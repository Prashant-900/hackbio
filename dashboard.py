"""
dashboard.py — Flask + Socket.IO live dashboard for the ABM simulation.

Run:  python dashboard.py
Open: http://localhost:5000
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter

import numpy as np
import yaml
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from agent import Phase
from simulate import Simulation
from visualize import generate_all_plots

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
sim_speed = 0.05  # seconds delay between epochs (lower = faster)

PHASE_INT = {Phase.LAG: 0, Phase.LOG: 1, Phase.STATIONARY: 2, Phase.DEATH: 3}


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _downsample(grid: np.ndarray, target: int = 100) -> list[list[float]]:
    h, w = grid.shape
    sh, sw = max(1, h // target), max(1, w // target)
    return np.round(grid[::sh, ::sw], 3).tolist()


def _bacteria_list(agents: list) -> list[list]:
    """Compact per-bacterium data for the world view."""
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
        ])
    return out


def build_snapshot(sim: Simulation) -> dict:
    alive = [a for a in sim.agents if a.alive]
    total_pop = len(alive)
    latest = sim.metrics[-1] if sim.metrics else {}

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
    total_toxin = np.zeros((sim.env.height, sim.env.width), dtype=np.float64)
    for grid in sim.env.toxin_grids.values():
        total_toxin += grid

    # Time-series
    ts = lambda key: [m[key] for m in sim.metrics]
    ts_get = lambda key, d=0: [m.get(key, d) for m in sim.metrics]

    # Per-genotype time series
    all_genos: set[int] = set()
    for m in sim.metrics:
        all_genos.update(m["genotype_counts"].keys())
    geno_ts = {}
    for g in sorted(all_genos):
        geno_ts[str(g)] = [m["genotype_counts"].get(g, 0) for m in sim.metrics]

    return {
        "epoch": sim.epoch,
        "total_epochs": sim.cfg["simulation"]["epochs"],
        "total_population": total_pop,
        "carrying_capacity": sim.cfg["population"]["carrying_capacity"],
        "grid_w": sim.env.width,
        "grid_h": sim.env.height,
        "mean_resource": round(sim.env.mean_resource(), 3),
        "mean_antibiotic": round(sim.env.mean_antibiotic(), 4),
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
        # Overlay grids
        "resource_grid": _downsample(sim.env.resource, 100),
        "antibiotic_grid": _downsample(sim.env.antibiotic, 100),
        "signal_grid": _downsample(sim.env.signal, 100),
        "biofilm_grid": _downsample(sim.env.biofilm.astype(np.float64), 100),
        "toxin_grid": _downsample(total_toxin, 100),
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
        "running": sim_running,
        "paused": sim_paused,
    }


# ── Simulation worker ──
def simulation_worker(cfg: dict):
    global sim, sim_running, sim_paused

    with sim_lock:
        sim = Simulation(cfg)
        sim_running = True
        sim_paused = False

    total_epochs = cfg["simulation"]["epochs"]
    socketio.emit("status", {"running": True, "paused": False, "epoch": 0})
    socketio.emit("snapshot", build_snapshot(sim))

    for ep in range(total_epochs):
        if not sim_running:
            break
        while sim_paused and sim_running:
            time.sleep(0.1)
        if not sim_running:
            break

        with sim_lock:
            sim.step()

        # Speed control delay
        if sim_speed > 0:
            time.sleep(sim_speed)

        if sim.epoch % update_interval == 0 or sim.epoch == 1 or sim.epoch == total_epochs:
            socketio.emit("snapshot", build_snapshot(sim))

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
            socketio.emit("snapshot", build_snapshot(sim))

    socketio.emit("status", {"running": False, "paused": False,
                             "epoch": sim.epoch if sim else 0})
    socketio.emit("log", {"msg": f"Simulation complete — {sim.epoch} epochs, "
                          f"{len(sim.agents)} agents alive"})


# ── Routes ──
@app.route("/")
def index():
    return render_template("index.html")


# ── Socket events ──
@socketio.on("connect")
def on_connect():
    if sim is not None:
        emit("snapshot", build_snapshot(sim))
    emit("status", {"running": sim_running, "paused": sim_paused,
                    "epoch": sim.epoch if sim else 0})
    emit("config_defaults", load_config())


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


@socketio.on("request_snapshot")
def on_request_snapshot():
    if sim is not None:
        emit("snapshot", build_snapshot(sim))


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
