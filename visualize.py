"""
visualize.py — Publication-quality charts saved to charts/ directory.

PS-mandated outputs:
  1. Evolution curves: population density vs. time grouped by genotype
  2. Resource dynamics: depletion curves matched with growth phases
  3. Spatial maps: colony density, biofilm clustering, genotype distribution,
     adversarial toxin boundaries
  4. Cooperation & competition indices over time
  5. Fitness landscape evolution
  6. Mutation frequency over time
  7. Phase distribution stacked area
"""

from __future__ import annotations

import os
from collections import Counter
from typing import TYPE_CHECKING, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

if TYPE_CHECKING:
    from simulate import Simulation

# ── Consistent style ────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.05)
GENOTYPE_CMAP = plt.cm.get_cmap("tab20", 20)
FIG_DPI = 180


def _get_genotype_color(g: int) -> Any:
    return GENOTYPE_CMAP(g % 20)


# ════════════════════════════════════════════════════════════
# 1. Population by Genotype (Evolution Curves)
# ════════════════════════════════════════════════════════════
def plot_population_by_genotype(metrics: list[dict], out_dir: str) -> str:
    epochs = [m["time_step"] for m in metrics]
    all_genos: set[int] = set()
    for m in metrics:
        all_genos.update(m["genotype_counts"].keys())
    sorted_g = sorted(all_genos)

    fig, ax = plt.subplots(figsize=(13, 6))
    for g in sorted_g:
        counts = [m["genotype_counts"].get(g, 0) for m in metrics]
        ax.plot(epochs, counts, linewidth=1.4, label=f"Genotype {g}",
                color=_get_genotype_color(g))
    ax.set_xlabel("Time Step (Epoch)")
    ax.set_ylabel("Population Count")
    ax.set_title("Population Density by Genotype over Time")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "evolution_curves.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ════════════════════════════════════════════════════════════
# 2. Resource Dynamics with Growth Phases
# ════════════════════════════════════════════════════════════
def plot_resource_dynamics(metrics: list[dict], out_dir: str) -> str:
    epochs = [m["time_step"] for m in metrics]
    pop = [m["total_population"] for m in metrics]
    res = [m["resource_concentration"] for m in metrics]
    ab = [m.get("mean_antibiotic", 0) for m in metrics]

    fig, ax1 = plt.subplots(figsize=(13, 6))
    color_pop, color_res, color_ab = "#2196F3", "#4CAF50", "#F44336"

    ax1.set_xlabel("Time Step (Epoch)")
    ax1.set_ylabel("Total Population", color=color_pop)
    ln1 = ax1.plot(epochs, pop, color=color_pop, linewidth=1.8, label="Population")
    ax1.tick_params(axis="y", labelcolor=color_pop)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Concentration", color=color_res)
    ln2 = ax2.plot(epochs, res, color=color_res, linewidth=1.5, ls="--", label="Resource")
    ln3 = ax2.plot(epochs, ab, color=color_ab, linewidth=1.5, ls=":", label="Antibiotic")
    ax2.tick_params(axis="y", labelcolor=color_res)

    # Phase annotations
    phase_lag = [m["phase_lag"] for m in metrics]
    phase_log = [m["phase_log"] for m in metrics]
    phase_stat = [m["phase_stationary"] for m in metrics]
    phase_death = [m["phase_death"] for m in metrics]

    # Find dominant phase per epoch
    for i, ep in enumerate(epochs):
        counts = {"Lag": phase_lag[i], "Log": phase_log[i],
                  "Stat": phase_stat[i], "Death": phase_death[i]}
        dominant = max(counts, key=counts.get)  # type: ignore
        if i % max(1, len(epochs) // 8) == 0 and pop[i] > 0:
            ax1.annotate(dominant, (ep, pop[i]), fontsize=7, alpha=0.7,
                         ha="center", va="bottom", color="#666")

    lns = ln1 + ln2 + ln3
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc="upper left", fontsize=9)
    ax1.set_title("Resource Dynamics & Bacterial Growth Phases")
    fig.tight_layout()

    path = os.path.join(out_dir, "resource_dynamics.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ════════════════════════════════════════════════════════════
# 3. Phase Distribution (Stacked Area)
# ════════════════════════════════════════════════════════════
def plot_phase_distribution(metrics: list[dict], out_dir: str) -> str:
    epochs = [m["time_step"] for m in metrics]
    lag = [m["phase_lag"] for m in metrics]
    log = [m["phase_log"] for m in metrics]
    stat = [m["phase_stationary"] for m in metrics]
    death = [m["phase_death"] for m in metrics]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.stackplot(epochs, lag, log, stat, death,
                 labels=["Lag", "Log", "Stationary", "Death"],
                 colors=["#78909c", "#66bb6a", "#ffa726", "#ef5350"], alpha=0.85)
    ax.set_xlabel("Time Step (Epoch)")
    ax.set_ylabel("Cell Count")
    ax.set_title("Growth Phase Distribution over Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    path = os.path.join(out_dir, "phase_distribution.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ════════════════════════════════════════════════════════════
# 4. Cooperation & Competition Indices
# ════════════════════════════════════════════════════════════
def plot_cooperation_competition(metrics: list[dict], out_dir: str) -> str:
    epochs = [m["time_step"] for m in metrics]
    coop = [m["cooperation_index"] for m in metrics]
    comp = [m["competition_index"] for m in metrics]
    bio = [m["biofilm_fraction"] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax1.plot(epochs, coop, color="#FF9800", linewidth=1.5, label="Cooperation Index")
    ax1.plot(epochs, bio, color="#8BC34A", linewidth=1.5, ls="--", label="Biofilm Fraction")
    ax1.set_ylabel("Cooperation")
    ax1.set_title("Cooperative & Adversarial Dynamics")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, comp, color="#F44336", linewidth=1.5, label="Competition Index (toxin)")
    ax2.set_xlabel("Time Step (Epoch)")
    ax2.set_ylabel("Competition")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    path = os.path.join(out_dir, "cooperation_competition.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ════════════════════════════════════════════════════════════
# 5. Fitness & Resistance Evolution
# ════════════════════════════════════════════════════════════
def plot_fitness_evolution(metrics: list[dict], out_dir: str) -> str:
    epochs = [m["time_step"] for m in metrics]
    fitness = [m["mean_fitness"] for m in metrics]
    resist = [m["mean_resistance"] for m in metrics]
    effic = [m["mean_efficiency"] for m in metrics]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(epochs, fitness, linewidth=1.8, label="Mean Fitness", color="#673AB7")
    ax.plot(epochs, resist, linewidth=1.5, ls="--", label="Mean Resistance", color="#E91E63")
    ax.plot(epochs, effic, linewidth=1.5, ls=":", label="Mean Efficiency", color="#009688")
    ax.set_xlabel("Time Step (Epoch)")
    ax.set_ylabel("Trait Value")
    ax.set_title("Fitness Landscape & Trait Evolution")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "fitness_evolution.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ════════════════════════════════════════════════════════════
# 6. Mutation Frequency & HGT Events
# ════════════════════════════════════════════════════════════
def plot_mutation_hgt(metrics: list[dict], out_dir: str) -> str:
    epochs = [m["time_step"] for m in metrics]
    mut_f = [m["mutation_frequency"] for m in metrics]
    hgt = [m["hgt_events"] for m in metrics]

    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax1.plot(epochs, mut_f, color="#FF5722", linewidth=1.5, label="Mutation Frequency")
    ax1.set_xlabel("Time Step (Epoch)")
    ax1.set_ylabel("Mutation Frequency", color="#FF5722")
    ax1.tick_params(axis="y", labelcolor="#FF5722")

    ax2 = ax1.twinx()
    ax2.bar(epochs, hgt, alpha=0.3, color="#3F51B5", label="HGT Events", width=1)
    ax2.set_ylabel("HGT Events", color="#3F51B5")
    ax2.tick_params(axis="y", labelcolor="#3F51B5")

    ax1.set_title("Mutation Frequency & Horizontal Gene Transfer Events")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    path = os.path.join(out_dir, "mutation_hgt.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ════════════════════════════════════════════════════════════
# 7. Spatial Maps (colony density, genotype, biofilm, toxin)
# ════════════════════════════════════════════════════════════
def _spatial_density_grid(sim: "Simulation") -> np.ndarray:
    grid = np.zeros((sim.env.height, sim.env.width), dtype=np.int32)
    for a in sim.agents:
        if a.alive:
            grid[a.y, a.x] += 1
    return grid


def _spatial_genotype_grid(sim: "Simulation") -> np.ndarray:
    grid = np.full((sim.env.height, sim.env.width), -1, dtype=np.int32)
    for a in sim.agents:
        if a.alive:
            grid[a.y, a.x] = a.genotype.id
    return grid


def _spatial_biofilm_grid(sim: "Simulation") -> np.ndarray:
    grid = np.zeros((sim.env.height, sim.env.width), dtype=np.int32)
    for a in sim.agents:
        if a.alive and a.biofilm_member:
            grid[a.y, a.x] += 1
    return grid


def plot_spatial_maps(sim: "Simulation", out_dir: str) -> list[str]:
    paths = []

    # Colony density
    density = _spatial_density_grid(sim)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(density, ax=ax, cmap="YlOrRd", cbar_kws={"label": "Cells per site"},
                square=True, xticklabels=False, yticklabels=False)
    ax.set_title(f"Colony Density — Epoch {sim.epoch}")
    p = os.path.join(out_dir, "spatial_colony_density.png")
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # Genotype distribution
    geno_grid = _spatial_genotype_grid(sim)
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.get_cmap("tab20", 20)
    masked = np.ma.masked_where(geno_grid < 0, geno_grid)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=19, interpolation="nearest", aspect="equal")
    plt.colorbar(im, ax=ax, label="Genotype ID", shrink=0.8)
    ax.set_title(f"Genotype Spatial Distribution — Epoch {sim.epoch}")
    ax.set_xticks([]); ax.set_yticks([])
    p = os.path.join(out_dir, "spatial_genotype_map.png")
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # Biofilm clustering
    biofilm_agents = _spatial_biofilm_grid(sim)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(biofilm_agents, ax=ax, cmap="GnBu", cbar_kws={"label": "Biofilm cells"},
                square=True, xticklabels=False, yticklabels=False)
    ax.set_title(f"Biofilm Clustering — Epoch {sim.epoch}")
    p = os.path.join(out_dir, "spatial_biofilm.png")
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # Resource heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(sim.env.resource, ax=ax, cmap="Greens",
                cbar_kws={"label": "Resource concentration"},
                square=True, xticklabels=False, yticklabels=False)
    ax.set_title(f"Resource Concentration — Epoch {sim.epoch}")
    p = os.path.join(out_dir, "spatial_resource.png")
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # Antibiotic heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(sim.env.antibiotic, ax=ax, cmap="Reds",
                cbar_kws={"label": "Antibiotic concentration"},
                square=True, xticklabels=False, yticklabels=False)
    ax.set_title(f"Antibiotic Concentration — Epoch {sim.epoch}")
    p = os.path.join(out_dir, "spatial_antibiotic.png")
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # Toxin adversarial boundary map (sum of all toxin grids)
    total_toxin = np.zeros((sim.env.height, sim.env.width), dtype=np.float64)
    for grid in sim.env.toxin_grids.values():
        total_toxin += grid
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(total_toxin, ax=ax, cmap="Purples",
                cbar_kws={"label": "Toxin concentration"},
                square=True, xticklabels=False, yticklabels=False)
    ax.set_title(f"Toxin (Bacteriocin) Landscape — Epoch {sim.epoch}")
    p = os.path.join(out_dir, "spatial_toxin_boundaries.png")
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # Quorum sensing signal
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(sim.env.signal, ax=ax, cmap="BuPu",
                cbar_kws={"label": "QS signal concentration"},
                square=True, xticklabels=False, yticklabels=False)
    ax.set_title(f"Quorum Sensing Signal — Epoch {sim.epoch}")
    p = os.path.join(out_dir, "spatial_quorum_signal.png")
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight"); plt.close(fig); paths.append(p)

    return paths


# ════════════════════════════════════════════════════════════
# 8. Births vs Deaths (demographic dynamics)
# ════════════════════════════════════════════════════════════
def plot_demographics(metrics: list[dict], out_dir: str) -> str:
    epochs = [m["time_step"] for m in metrics]
    divs = [m["divisions"] for m in metrics]
    deaths = [m["deaths"] for m in metrics]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(epochs, divs, alpha=0.4, color="#4CAF50", label="Divisions (births)")
    ax.fill_between(epochs, deaths, alpha=0.4, color="#F44336", label="Deaths")
    ax.plot(epochs, divs, color="#4CAF50", linewidth=1)
    ax.plot(epochs, deaths, color="#F44336", linewidth=1)
    ax.set_xlabel("Time Step (Epoch)")
    ax.set_ylabel("Events per Epoch")
    ax.set_title("Demographic Dynamics: Births vs Deaths")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "demographics.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ════════════════════════════════════════════════════════════
# 9. Genotype Density Relative Frequency (stacked)
# ════════════════════════════════════════════════════════════
def plot_genotype_frequency(metrics: list[dict], out_dir: str) -> str:
    epochs = [m["time_step"] for m in metrics]
    all_genos: set[int] = set()
    for m in metrics:
        all_genos.update(m["genotype_density"].keys())
    sorted_g = sorted(all_genos)

    freq_data = []
    for g in sorted_g:
        freq_data.append([m["genotype_density"].get(g, 0.0) for m in metrics])

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = [_get_genotype_color(g) for g in sorted_g]
    labels = [f"Geno {g}" for g in sorted_g]
    ax.stackplot(epochs, *freq_data, labels=labels, colors=colors, alpha=0.85)
    ax.set_xlabel("Time Step (Epoch)")
    ax.set_ylabel("Relative Frequency")
    ax.set_title("Genotype Density (Relative Frequency) over Time")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)
    path = os.path.join(out_dir, "genotype_frequency.png")
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ════════════════════════════════════════════════════════════
# Master function
# ════════════════════════════════════════════════════════════
def generate_all_plots(sim: "Simulation") -> list[str]:
    """Generate all PS-required charts and spatial maps."""
    charts_dir = sim.cfg["simulation"].get("charts_dir", "charts")
    os.makedirs(charts_dir, exist_ok=True)

    paths = [
        plot_population_by_genotype(sim.metrics, charts_dir),
        plot_genotype_frequency(sim.metrics, charts_dir),
        plot_resource_dynamics(sim.metrics, charts_dir),
        plot_phase_distribution(sim.metrics, charts_dir),
        plot_cooperation_competition(sim.metrics, charts_dir),
        plot_fitness_evolution(sim.metrics, charts_dir),
        plot_mutation_hgt(sim.metrics, charts_dir),
        plot_demographics(sim.metrics, charts_dir),
    ]
    paths += plot_spatial_maps(sim, charts_dir)
    return paths
