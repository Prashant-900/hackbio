"""
environment.py — Spatial grids for resources, antibiotics, QS signals, toxins,
                  and biofilm EPS matrix.

Biological basis:
  - Resource diffusion: Fick's second law (discretised 2D mean-filter)
    with Neumann (no-flux) boundary conditions (reflects mass at edges).
  - Resource depletion: Glucose-limited chemostat model (cf. DM25 in LTEE,
    Lenski 2010). Replenishment rate models continuous-flow reactor.
  - Antibiotic: first-order decay + spatial diffusion; introduced as a
    gradient from the top edge (disk-diffusion assay analogy).
  - QS signal: AHL autoinducer with enzymatic decay (AiiA lactonase)
  - Toxin: per-genotype bacteriocin grids
  - Biofilm: EPS matrix grid — collective public good

Conservation law: Resources are finite. Consumption by agents removes from grid.
Replenishment only in "resource-rich" mode (models chemostat/flow reactor).

LTEE context:
  - The E. coli LTEE (Lenski et al., 1988–present) uses DM25 minimal glucose
    medium (25 mg/L glucose), supporting ~5×10^8 cells in 10 mL.
  - Daily 1:100 serial transfer gives 6.64 generations/day.
  - Our per-epoch replenishment models the nutrient refresh aspect of
    serial passage without explicit dilution events.
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

        # ── Physics (Cardinal models — Rosso et al. 1993, 1995) ──
        physics = cfg.get("physics", {})
        self.temperature: float = physics.get("temperature", 37.0)
        self.pressure_atm: float = physics.get("pressure_atm", 1.0)
        self.ph: float = physics.get("ph", 7.0)
        self.z_levels: int = cfg.get("grid", {}).get("z_levels", 10)
        self.growth_modifier: float = 1.0  # recomputed each step()

    # ──────────────────────────────────────────────────────────────
    # Diffusion (Fick's 2nd law, discretised)
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _diffuse(grid: np.ndarray, rate: float) -> np.ndarray:
        # Neumann (no-flux) boundaries — reflect mode preserves mass at edges
        blurred = uniform_filter(grid, size=3, mode="reflect")
        return grid + rate * (blurred - grid)

    # ──────────────────────────────────────────────────────────────
    # Per-epoch environmental update
    # ──────────────────────────────────────────────────────────────
    def step(self, epoch: int) -> None:
        self.growth_modifier = (
            self.temperature_growth_factor(self.temperature)
            * self.ph_growth_factor(self.ph)
            * self.pressure_growth_factor(self.pressure_atm)
        )
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
            elapsed = epoch - self.ab_start_epoch

            if self.ab_mode == "gradual":
                # Advancing front: the injection row moves downward each epoch.
                # The front advances at ~1 row/epoch, so the full grid is
                # covered over H epochs — models IV drip or gradual perfusion.
                front_row = min(elapsed, self.height - 1)
                # Inject at the current front row and a few rows above it
                band = max(1, int(self.height * 0.05))  # 5% of grid as band
                for r in range(max(0, front_row - band), front_row + 1):
                    self.antibiotic[r, :] += self.ab_gradual_rate

            elif self.ab_mode == "spike":
                # Single bolus applied uniformly at start epoch
                if epoch == self.ab_start_epoch:
                    self.antibiotic += self.ab_spike_conc

            elif self.ab_mode == "center":
                # Radial spread from center — models localized injection.
                # The radius grows each epoch; concentration decays with
                # distance from center (Gaussian-like profile).
                cy, cx = self.height / 2.0, self.width / 2.0
                radius = min(elapsed * 1.0, max(self.height, self.width) / 2.0)
                yy, xx = np.ogrid[:self.height, :self.width]
                dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                mask = dist <= radius
                # Concentration falls off with distance from center
                falloff = np.where(mask, 1.0 - dist / (radius + 1e-9), 0.0)
                self.antibiotic += self.ab_gradual_rate * falloff

            elif self.ab_mode == "sweep":
                # Full-width sweep from top to bottom over grid-height epochs.
                # Each epoch, a thin horizontal band injects antibiotic.
                row = min(elapsed, self.height - 1)
                self.antibiotic[row, :] += self.ab_gradual_rate * 2.0

        self.antibiotic *= (1.0 - self.ab_decay)
        self.antibiotic = self._diffuse(self.antibiotic, self.ab_diffusion)
        ab_max = self.cfg.get("antibiotic", {}).get("max_concentration", 10.0)
        np.clip(self.antibiotic, 0.0, ab_max, out=self.antibiotic)

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

    def ensure_genotype_toxin_grid(self, genotype_id: int) -> None:
        if genotype_id not in self.toxin_grids:
            self.toxin_grids[genotype_id] = np.zeros(self.shape, dtype=np.float64)

    def cleanup_toxin_grids(self, active_genotypes: set[int]) -> None:
        for g in list(self.toxin_grids):
            if g not in active_genotypes:
                del self.toxin_grids[g]

    # ──────────────────────────────────────────────────────────────
    # Physics growth-factor models
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def temperature_growth_factor(
        T: float, T_min: float = 10.0, T_opt: float = 37.0, T_max: float = 45.0
    ) -> float:
        """Cardinal Temperature Model with Inflection (Rosso et al. 1993).

        Returns a factor in [0, 1].  At T_opt the factor is 1.0;
        at T ≤ T_min or T ≥ T_max the factor is 0.0.
        """
        if T <= T_min or T >= T_max:
            return 0.0
        num = (T - T_max) * (T - T_min) ** 2
        den = (T_opt - T_min) * (
            (T_opt - T_min) * (T - T_opt)
            - (T_opt - T_max) * (T_opt + T_min - 2 * T)
        )
        return max(0.0, min(1.0, num / den))

    @staticmethod
    def ph_growth_factor(
        pH: float, pH_min: float = 4.0, pH_opt: float = 7.0, pH_max: float = 9.0
    ) -> float:
        """Cardinal pH Model (Rosso et al. 1995).

        Analogous to the CTMI but for pH.  Factor = 1 at pH_opt.
        """
        if pH <= pH_min or pH >= pH_max:
            return 0.0
        denom = (pH - pH_min) * (pH - pH_max) - (pH - pH_opt) ** 2
        if abs(denom) < 1e-12:
            return 0.0
        return max(0.0, min(1.0, ((pH - pH_min) * (pH - pH_max)) / denom))

    @staticmethod
    def pressure_growth_factor(P_atm: float) -> float:
        """Pressure effect on E. coli growth (Abe & Horikoshi 2001).

        Growth rate declines linearly above 1 atm.  Full inhibition
        at ~500 atm (deep-sea pressures).
        """
        if P_atm <= 1.0:
            return 1.0
        return max(0.0, 1.0 - (P_atm - 1.0) / 500.0)

    def mean_resource(self) -> float:
        return float(np.mean(self.resource))

    def total_resource(self) -> float:
        return float(np.sum(self.resource))

    def mean_antibiotic(self) -> float:
        return float(np.mean(self.antibiotic))

    def mean_biofilm(self) -> float:
        return float(np.mean(self.biofilm))


