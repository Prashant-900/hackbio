"""
environment.py — Spatial grids for resources, antibiotics, QS signals, toxins,
                  and biofilm EPS matrix.

Biological basis:
  - Resource diffusion: Fick's second law (discretised 2D mean-filter)
  - Antibiotic: first-order decay + spatial diffusion
  - QS signal: AHL autoinducer with enzymatic decay (AiiA lactonase)
  - Toxin: per-genotype bacteriocin grids
  - Biofilm: EPS matrix grid — collective public good

Conservation law: Resources are finite. Consumption by agents removes from grid.
Replenishment only in "resource-rich" mode (models chemostat/flow reactor).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


class Environment:
    """2-D chemical landscape of the simulation."""

    def __init__(self, cfg: dict) -> None:
        self.width: int = cfg["grid"]["width"]
        self.height: int = cfg["grid"]["height"]
        self.shape = (self.height, self.width)
        self.cfg = cfg

        # ── Resource grid ──
        res_cfg = cfg["resource"]
        self.resource_scenario: str = res_cfg["scenario"]
        self.resource: np.ndarray = np.full(
            self.shape, res_cfg["initial_concentration"], dtype=np.float64
        )
        self.resource_replenishment: float = res_cfg["replenishment_rate"]
        self.resource_diffusion: float = res_cfg["diffusion_rate"]
        self.resource_max: float = res_cfg.get("max_concentration", 20.0)

        # Track cumulative resource consumed (mass-balance bookkeeping)
        self.total_resource_consumed: float = 0.0

        # ── Antibiotic grid ──
        ab_cfg = cfg["antibiotic"]
        self.antibiotic: np.ndarray = np.zeros(self.shape, dtype=np.float64)
        self.ab_mode: str = ab_cfg["mode"]
        self.ab_start_epoch: int = ab_cfg["start_epoch"]
        self.ab_gradual_rate: float = ab_cfg["gradual_rate"]
        self.ab_spike_conc: float = ab_cfg["spike_concentration"]
        self.ab_decay: float = ab_cfg["decay_rate"]
        self.ab_diffusion: float = ab_cfg["diffusion_rate"]

        # ── Quorum-sensing signal grid ──
        qs_cfg = cfg["quorum_sensing"]
        self.signal: np.ndarray = np.zeros(self.shape, dtype=np.float64)
        self.signal_diffusion: float = qs_cfg["signal_diffusion_rate"]
        self.signal_decay: float = qs_cfg["signal_decay_rate"]

        # ── Biofilm EPS matrix grid ──
        self.biofilm: np.ndarray = np.zeros(self.shape, dtype=np.float64)
        self.biofilm_decay: float = 0.02  # slow degradation of EPS

        # ── Per-genotype toxin grids ──
        n_genotypes = cfg["genotype"]["initial_types"]
        self.toxin_grids: dict[int, np.ndarray] = {
            g: np.zeros(self.shape, dtype=np.float64) for g in range(n_genotypes)
        }
        tox_cfg = cfg["toxin"]
        self.toxin_diffusion: float = tox_cfg["diffusion_rate"]
        self.toxin_decay: float = tox_cfg["decay_rate"]

    # ──────────────────────────────────────────────────────────────
    # Diffusion (Fick's 2nd law, discretised)
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _diffuse(grid: np.ndarray, rate: float) -> np.ndarray:
        blurred = uniform_filter(grid, size=3, mode="constant", cval=0.0)
        return grid + rate * (blurred - grid)

    # ──────────────────────────────────────────────────────────────
    # Per-epoch environmental update
    # ──────────────────────────────────────────────────────────────
    def step(self, epoch: int) -> None:
        self._update_resources()
        self._update_antibiotic(epoch)
        self._update_signal()
        self._update_biofilm()
        self._update_toxins()

    def _update_resources(self) -> None:
        self.resource = self._diffuse(self.resource, self.resource_diffusion)
        if self.resource_scenario == "resource-rich":
            self.resource += self.resource_replenishment
        np.clip(self.resource, 0.0, self.resource_max, out=self.resource)

    def _update_antibiotic(self, epoch: int) -> None:
        if epoch >= self.ab_start_epoch:
            if self.ab_mode == "gradual":
                self.antibiotic += self.ab_gradual_rate
            elif self.ab_mode == "spike" and epoch == self.ab_start_epoch:
                self.antibiotic += self.ab_spike_conc
        self.antibiotic *= (1.0 - self.ab_decay)
        self.antibiotic = self._diffuse(self.antibiotic, self.ab_diffusion)
        np.clip(self.antibiotic, 0.0, None, out=self.antibiotic)

    def _update_signal(self) -> None:
        self.signal *= (1.0 - self.signal_decay)
        self.signal = self._diffuse(self.signal, self.signal_diffusion)
        np.clip(self.signal, 0.0, None, out=self.signal)

    def _update_biofilm(self) -> None:
        self.biofilm *= (1.0 - self.biofilm_decay)
        self.biofilm = self._diffuse(self.biofilm, 0.02)  # minimal EPS spread
        np.clip(self.biofilm, 0.0, None, out=self.biofilm)

    def _update_toxins(self) -> None:
        for g in list(self.toxin_grids):
            grid = self.toxin_grids[g]
            grid *= (1.0 - self.toxin_decay)
            self.toxin_grids[g] = self._diffuse(grid, self.toxin_diffusion)
            np.clip(self.toxin_grids[g], 0.0, None, out=self.toxin_grids[g])

    # ──────────────────────────────────────────────────────────────
    # Agent interaction helpers
    # ──────────────────────────────────────────────────────────────
    def consume_resource(self, x: int, y: int, amount: float) -> float:
        available = self.resource[y, x]
        consumed = min(available, max(0.0, amount))
        self.resource[y, x] -= consumed
        self.total_resource_consumed += consumed
        return consumed

    def add_signal(self, x: int, y: int, amount: float) -> None:
        self.signal[y, x] += amount

    def add_biofilm(self, x: int, y: int, amount: float) -> None:
        self.biofilm[y, x] += amount

    def add_toxin(self, genotype: int, x: int, y: int, amount: float) -> None:
        if genotype not in self.toxin_grids:
            self.toxin_grids[genotype] = np.zeros(self.shape, dtype=np.float64)
        self.toxin_grids[genotype][y, x] += amount

    def get_foreign_toxin(self, own_genotype: int, x: int, y: int) -> float:
        total = 0.0
        for g, grid in self.toxin_grids.items():
            if g != own_genotype:
                total += grid[y, x]
        return total

    def get_total_toxin_at(self, x: int, y: int) -> float:
        return sum(grid[y, x] for grid in self.toxin_grids.values())

    def mean_resource(self) -> float:
        return float(np.mean(self.resource))

    def total_resource(self) -> float:
        return float(np.sum(self.resource))

    def mean_antibiotic(self) -> float:
        return float(np.mean(self.antibiotic))

    def mean_biofilm(self) -> float:
        return float(np.mean(self.biofilm))

    def ensure_genotype_toxin_grid(self, genotype: int) -> None:
        if genotype not in self.toxin_grids:
            self.toxin_grids[genotype] = np.zeros(self.shape, dtype=np.float64)

    def cleanup_toxin_grids(self, active_genotypes: set[int]) -> None:
        for g in list(self.toxin_grids.keys()):
            if g not in active_genotypes:
                del self.toxin_grids[g]
