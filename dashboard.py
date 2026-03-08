"""
dashboard.py — Flask + Socket.IO live dashboard for the ABM simulation.

Run:  python dashboard.py
Open: http://localhost:5000
"""

from __future__ import annotations

import glob
import io
import json
import os
import tempfile
import threading
import time
import zipfile
from collections import Counter

import numpy as np
import yaml
from flask import Flask, jsonify, render_template, send_file
from flask_socketio import SocketIO, emit

from agent import Phase
from simulate import Simulation
from visualize import generate_all_plots

try:
    from gpu_utils import gpu_info as _gpu_info
    GPU_UTILS_OK = True
except ImportError:
    GPU_UTILS_OK = False
    def _gpu_info():
        return {"cuda_available": False, "mps_available": False,
                "device": "cpu", "gpu_name": None,
                "gpu_memory_mb": None, "torch_version": "N/A"}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "hackbio-abm-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Global state ──
sim: Simulation | None = None
sim_thread: threading.Thread | None = None
sim_lock = threading.Lock()
sim_running = False
sim_paused = False
update_interval = 3
sim_speed = 0.0  # default: no delay (fastest)
sim_view_mode = "2d"  # '2d' or '3d' — affects z-axis movement

# Decoupled rendering: sim stores latest snapshot, frontend pulls on its own pace
_latest_snapshot: dict | None = None
_snapshot_lock = threading.Lock()
_snapshot_epoch: int = -1  # epoch of latest cached snapshot

PHASE_INT = {Phase.LAG: 0, Phase.LOG: 1, Phase.STATIONARY: 2, Phase.DEATH: 3}


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _downsample(grid: np.ndarray, target: int = 100) -> list[list[float]]:
    h, w = grid.shape
    sh, sw = max(1, h // target), max(1, w // target)
    return np.round(grid[::sh, ::sw], 3).tolist()


def _bacteria_list(agents: list) -> list[list]:
    """Compact per-bacterium data for the world view — optimized."""
    out = []
    for a in agents:
        if not a.alive:
            continue
        out.append([
            a.x, a.y,
            a.genotype.id,
            PHASE_INT.get(a.phase, 0),
            round(a.fitness, 3),
            round(a.biomass, 3),
            round(a.genotype.antibiotic_resistance, 3),
            round(a.genotype.nutrient_efficiency, 3),
            round(a.genotype.toxin_production, 3),
            round(a.genotype.public_good_production, 3),
            int(a.biofilm_member),
            a.age,
            round(getattr(a, 'z', 0.0), 2),
        ])
    return out


def build_snapshot(sim_obj: Simulation) -> dict:
    alive = [a for a in sim_obj.agents if a.alive]
    total_pop = len(alive)
    latest = sim_obj.metrics[-1] if sim_obj.metrics else {}

    geno_counts = dict(Counter(a.genotype.id for a in alive))
    phase_counts = dict(Counter(a.phase.name for a in alive))
    n_biofilm = sum(1 for a in alive if a.biofilm_member)

    if total_pop > 0:
        mean_resistance = round(float(np.mean([a.genotype.antibiotic_resistance for a in alive])), 4)
        mean_efficiency = round(float(np.mean([a.genotype.nutrient_efficiency for a in alive])), 4)
        mean_fitness = round(float(np.mean([a.fitness for a in alive])), 4)
    else:
        mean_resistance = mean_efficiency = mean_fitness = 0.0

    # Overlay grids (downsampled for field visualization)
    total_toxin = np.zeros((sim_obj.env.height, sim_obj.env.width), dtype=np.float64)
    for grid in sim_obj.env.toxin_grids.values():
        total_toxin += grid

    # Time-series
    ts = lambda key: [m[key] for m in sim_obj.metrics]
    ts_get = lambda key, d=0: [m.get(key, d) for m in sim_obj.metrics]

    # Per-genotype time series
    all_genos: set[int] = set()
    for m in sim_obj.metrics:
        all_genos.update(m["genotype_counts"].keys())
    geno_ts = {}
    for g in sorted(all_genos):
        geno_ts[str(g)] = [m["genotype_counts"].get(g, 0) for m in sim_obj.metrics]

    return {
        "epoch": sim_obj.epoch,
        "total_epochs": sim_obj.cfg["simulation"]["epochs"],
        "total_population": total_pop,
        "carrying_capacity": sim_obj.cfg["population"]["carrying_capacity"],
        "grid_w": sim_obj.env.width,
        "grid_h": sim_obj.env.height,
        "mean_resource": round(sim_obj.env.mean_resource(), 3),
        "mean_antibiotic": round(sim_obj.env.mean_antibiotic(), 4),
        "genotype_counts": geno_counts,
        "phase_counts": phase_counts,
        "cooperation_index": latest.get("cooperation_index", 0),
        "competition_index": latest.get("competition_index", 0),
        "biofilm_count": n_biofilm,
        "biofilm_fraction": latest.get("biofilm_fraction", 0),
        "mean_resistance": mean_resistance,
        "mean_efficiency": mean_efficiency,
        "mean_fitness": mean_fitness,
        "mutation_frequency": latest.get("mutation_frequency", 0),
        "hgt_events": latest.get("hgt_events", 0),
        "divisions": latest.get("divisions", 0),
        "deaths": latest.get("deaths", 0),
        # Per-bacterium data for world view
        "bacteria": _bacteria_list(alive),
        # Overlay grids (downsampled for smaller payloads)
        "resource_grid": _downsample(sim_obj.env.resource, 80),
        "antibiotic_grid": _downsample(sim_obj.env.antibiotic, 80),
        "signal_grid": _downsample(sim_obj.env.signal, 80),
        "biofilm_grid": _downsample(sim_obj.env.biofilm.astype(np.float64), 80),
        "toxin_grid": _downsample(total_toxin, 80),
        # Time-series
        "ts_epochs": ts("time_step"),
        "ts_population": ts("total_population"),
        "ts_resource": ts("resource_concentration"),
        "ts_cooperation": ts("cooperation_index"),
        "ts_competition": ts("competition_index"),
        "ts_fitness": ts("mean_fitness"),
        "ts_resistance": ts("mean_resistance"),
        "ts_mutation": ts("mutation_frequency"),
        "ts_biofilm": ts("biofilm_fraction"),
        "ts_hgt": ts("hgt_events"),
        "ts_divisions": ts("divisions"),
        "ts_deaths": ts("deaths"),
        "ts_antibiotic": ts_get("mean_antibiotic", 0),
        "ts_phase_lag": ts("phase_lag"),
        "ts_phase_log": ts("phase_log"),
        "ts_phase_stat": ts("phase_stationary"),
        "ts_phase_death": ts("phase_death"),
        "ts_genotypes": geno_ts,
        "ts_cumulative_mutations": ts_get("cumulative_mutations", 0),
        "ts_cumulative_hgt": ts_get("cumulative_hgt", 0),
        "ts_resource_consumed": ts_get("total_resource_consumed", 0),
        "running": sim_running,
        "paused": sim_paused,
        # Physics
        "temperature": getattr(sim_obj.env, 'temperature', 37.0),
        "pressure_atm": getattr(sim_obj.env, 'pressure_atm', 1.0),
        "ph": getattr(sim_obj.env, 'ph', 7.0),
        "growth_modifier": round(getattr(sim_obj.env, 'growth_modifier', 1.0), 4),
        # RL stats
        "rl_stats": sim_obj.dqn.stats() if getattr(sim_obj, 'dqn', None) else None,
    }


# ── Snapshot file path ──
SNAPSHOT_FILE = os.path.join("output", "latest_state.json")


# ── Simulation worker (fully decoupled: compute → write file) ──
def _write_snapshot():
    """Build snapshot, cache in memory, and write to file atomically.
    The sim thread calls this; the UI reads the file via HTTP."""
    global _latest_snapshot, _snapshot_epoch
    snap = build_snapshot(sim)
    with _snapshot_lock:
        _latest_snapshot = snap
        _snapshot_epoch = sim.epoch
    # Atomic write: write to temp, then rename (no partial reads)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir="output", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(snap, f, separators=(",", ":"))
        os.replace(tmp, SNAPSHOT_FILE)
    except Exception:
        # If file write fails, in-memory cache still works
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def simulation_worker(cfg: dict):
    global sim, sim_running, sim_paused, _latest_snapshot, _snapshot_epoch

    # Inject view mode into config so agents can check it during division
    cfg['_view_mode'] = sim_view_mode

    with sim_lock:
        sim = Simulation(cfg)
        sim.view_mode = sim_view_mode
        sim_running = True
        sim_paused = False

    total_epochs = cfg["simulation"]["epochs"]
    socketio.emit("status", {"running": True, "paused": False, "epoch": 0})
    _write_snapshot()

    for ep in range(total_epochs):
        if not sim_running:
            break
        while sim_paused and sim_running:
            time.sleep(0.1)
        if not sim_running:
            break

        with sim_lock:
            sim.step()

        # Optional speed throttle (0 = no delay = fastest)
        if sim_speed > 0:
            time.sleep(sim_speed)

        # Write snapshot to file — UI fetches independently via /api/snapshot
        if sim.epoch % update_interval == 0 or sim.epoch == 1 or sim.epoch == total_epochs:
            _write_snapshot()
            # Lightweight epoch counter so UI knows to fetch
            socketio.emit("epoch_tick", {"epoch": sim.epoch, "total": total_epochs})

        if len(sim.agents) == 0:
            socketio.emit("log", {"msg": f"Population extinct at epoch {sim.epoch}"})
            break

    with sim_lock:
        sim_running = False
        if sim:
            sim.export_csv()
            try:
                generate_all_plots(sim)
                socketio.emit("log", {"msg": "Charts saved to charts/ directory."})
            except Exception as e:
                socketio.emit("log", {"msg": f"Chart generation error: {e}"})
            _write_snapshot()

    socketio.emit("status", {"running": False, "paused": False,
                             "epoch": sim.epoch if sim else 0})
    socketio.emit("sim_complete", {})
    socketio.emit("log", {"msg": f"Simulation complete — {sim.epoch} epochs, "
                          f"{len(sim.agents)} agents alive"})


# ── Routes ──
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/snapshot")
def api_snapshot():
    """HTTP endpoint: frontend polls this to get latest state.
    Serves from memory cache (fast), falls back to file."""
    with _snapshot_lock:
        snap = _latest_snapshot
    if snap is not None:
        return jsonify(snap)
    # Fallback: read from file
    if os.path.isfile(SNAPSHOT_FILE):
        try:
            with open(SNAPSHOT_FILE, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return jsonify({}), 204


@app.route("/report")
def download_report():
    """Generate a ZIP containing all chart PNGs and the CSV metrics file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add chart images
        for png in sorted(glob.glob("charts/*.png")):
            zf.write(png, os.path.join("report", png))
        # Add CSV
        csv_path = os.path.join("output", "simulation_metrics.csv")
        if os.path.isfile(csv_path):
            zf.write(csv_path, os.path.join("report", "simulation_metrics.csv"))
        # Add config file
        if os.path.isfile("config.yaml"):
            zf.write("config.yaml", os.path.join("report", "config.yaml"))
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name="simulation_report.zip",
    )


# ── Socket events ──
@socketio.on("connect")
def on_connect():
    # No snapshot push — frontend polls /api/snapshot via HTTP
    emit("status", {"running": sim_running, "paused": sim_paused,
                    "epoch": sim.epoch if sim else 0})
    emit("config_defaults", load_config())
    emit("gpu_info", _gpu_info())


@socketio.on("start")
def on_start(data=None):
    global sim_thread, sim_running, sim_paused, update_interval, sim_speed

    if sim_running:
        emit("log", {"msg": "Simulation already running."})
        return

    cfg = load_config()
    if data:
        mapping = {
            "epochs": ("simulation", "epochs", int),
            "initial_count": ("bacterium", "initial_count", int),
            "carrying_capacity": ("population", "carrying_capacity", int),
            "resource_scenario": ("resource", "scenario", str),
            "antibiotic_mode": ("antibiotic", "mode", str),
            "antibiotic_start": ("antibiotic", "start_epoch", int),
            "mutation_rate": ("mutation", "rate", float),
            "grid_width": ("grid", "width", int),
            "grid_height": ("grid", "height", int),
            "z_levels": ("grid", "z_levels", int),
        }
        for key, (sec, param, typ) in mapping.items():
            if key in data and data[key] not in (None, ""):
                cfg[sec][param] = typ(data[key])
        if "update_interval" in data:
            update_interval = max(1, int(data["update_interval"]))
        if "speed" in data:
            sim_speed = max(0.0, float(data["speed"]))
        if "seed" in data:
            val = data["seed"]
            cfg["simulation"]["seed"] = int(val) if val not in (None, "", "null") else None
        # Physics settings from UI
        if "temperature" in data and data["temperature"] not in (None, ""):
            cfg.setdefault("physics", {})["temperature"] = float(data["temperature"])
        if "pressure_atm" in data and data["pressure_atm"] not in (None, ""):
            cfg.setdefault("physics", {})["pressure_atm"] = float(data["pressure_atm"])
        if "ph" in data and data["ph"] not in (None, ""):
            cfg.setdefault("physics", {})["ph"] = float(data["ph"])
        # RL toggle
        if "rl_enabled" in data:
            cfg.setdefault("rl", {})["enabled"] = bool(data["rl_enabled"])
        if "force_cpu" in data:
            cfg.setdefault("rl", {})["force_cpu"] = bool(data["force_cpu"])
        if "view_mode" in data:
            global sim_view_mode
            sim_view_mode = str(data["view_mode"])

    sim_thread = threading.Thread(target=simulation_worker, args=(cfg,), daemon=True)
    sim_thread.start()
    emit("log", {"msg": "Simulation started."})


@socketio.on("pause")
def on_pause():
    global sim_paused
    if sim_running:
        sim_paused = not sim_paused
        socketio.emit("status", {"running": sim_running, "paused": sim_paused,
                                 "epoch": sim.epoch if sim else 0})
        socketio.emit("log", {"msg": "Paused." if sim_paused else "Resumed."})


@socketio.on("stop")
def on_stop():
    global sim_running, sim_paused
    sim_running = False
    sim_paused = False
    socketio.emit("status", {"running": False, "paused": False,
                             "epoch": sim.epoch if sim else 0})
    socketio.emit("log", {"msg": "Simulation stopped."})


@socketio.on("set_speed")
def on_set_speed(data):
    global sim_speed
    sim_speed = max(0.0, float(data.get("value", 0.05)))
    emit("log", {"msg": f"Speed delay: {sim_speed:.3f}s/epoch"})


@socketio.on("set_update_interval")
def on_set_interval(data):
    global update_interval
    update_interval = max(1, int(data.get("value", 3)))
    emit("log", {"msg": f"Update interval: every {update_interval} epochs."})


@socketio.on("set_view_mode")
def on_set_view_mode(data):
    """Toggle 2D/3D mode — affects bacterial z-axis movement."""
    global sim_view_mode
    mode = data.get("mode", "2d")
    sim_view_mode = mode
    if sim is not None:
        sim.view_mode = mode
    emit("log", {"msg": f"View mode: {mode.upper()} — {'3-axis' if mode=='3d' else '2-axis'} movement"})


@socketio.on("request_snapshot")
def on_request_snapshot():
    """Legacy: clients can still request via socket; serves from memory cache."""
    with _snapshot_lock:
        snap = _latest_snapshot
    if snap is not None:
        emit("snapshot", snap)


# ── Main ──
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    os.makedirs("charts", exist_ok=True)
    print("=" * 60)
    print("  Bacterial Colony ABM — Live Dashboard")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False,
                 allow_unsafe_werkzeug=True)
