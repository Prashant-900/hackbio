"""
environment.py — Spatial grids for resources, antibiotics, QS signals, toxins,
                  and biofilm EPS matrix.

GPU acceleration:
  When a CUDA or MPS device is available, all grids are stored as torch
  tensors on-device.  Diffusion, decay, injection, and clipping run
  entirely on GPU via TensorBackend (torch conv2d + element-wise ops).
  Numpy views are materialised lazily for per-agent scalar reads (grid[y,x])
  and are invalidated each epoch.

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
"""

from __future__ import annotations

import numpy as np

try:
    from gpu_utils import TensorBackend
    _TB_OK = True
except ImportError:
    _TB_OK = False

# Fallback: scipy only needed for CPU path
try:
    from scipy.ndimage import uniform_filter
except ImportError:
    uniform_filter = None  # type: ignore[assignment]


class Environment:
    """2-D chemical landscape of the simulation.

    If a GPU is available the heavy per-epoch work (diffusion, decay,
    replenishment, clipping) runs entirely on-device.  Agent-level
    scalar accesses (consume_resource, add_signal, …) operate on
    synced numpy views that are refreshed once per epoch.
    """

    def __init__(self, cfg: dict, force_cpu: bool = False) -> None:
        self.width: int = cfg["grid"]["width"]
        self.height: int = cfg["grid"]["height"]
        self.shape = (self.height, self.width)
        self.cfg = cfg

        # ── Tensor backend (GPU when available) ──
        self._tb: TensorBackend | None = None
        self._use_gpu: bool = False
        if _TB_OK:
            self._tb = TensorBackend(force_cpu=force_cpu)
            self._use_gpu = self._tb.is_gpu

        # ── Resource grid ──
        res_cfg = cfg["resource"]
        self.resource_scenario: str = res_cfg["scenario"]
        self.resource: np.ndarray = np.full(
            self.shape, res_cfg["initial_concentration"], dtype=np.float64
        )
        self.resource_replenishment: float = res_cfg["replenishment_rate"]
        self.resource_diffusion: float = res_cfg["diffusion_rate"]
        self.resource_max: float = res_cfg.get("max_concentration", 20.0)

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
        self.biofilm_decay: float = 0.02

        # ── Per-genotype toxin grids ──
        n_genotypes = cfg["genotype"]["initial_types"]
        self.toxin_grids: dict[int, np.ndarray] = {
            g: np.zeros(self.shape, dtype=np.float64) for g in range(n_genotypes)
        }
        tox_cfg = cfg["toxin"]
        self.toxin_diffusion: float = tox_cfg["diffusion_rate"]
        self.toxin_decay: float = tox_cfg["decay_rate"]

        # ── Physics ──
        physics = cfg.get("physics", {})
        self.temperature: float = physics.get("temperature", 37.0)
        self.pressure_atm: float = physics.get("pressure_atm", 1.0)
        self.ph: float = physics.get("ph", 7.0)
        self.z_levels: int = cfg.get("grid", {}).get("z_levels", 10)
        self.growth_modifier: float = 1.0

    # ──────────────────────────────────────────────────────────
    # GPU helpers
    # ──────────────────────────────────────────────────────────
    def _to_tensor(self, arr: np.ndarray):
        """Convert numpy grid to on-device tensor."""
        return self._tb.from_numpy(arr)

    def _from_tensor(self, t) -> np.ndarray:
        """Convert on-device tensor back to float64 numpy."""
        return TensorBackend.to_numpy(t)

    # ──────────────────────────────────────────────────────────
    # Diffusion — CPU fallback (scipy uniform_filter)
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _diffuse_cpu(grid: np.ndarray, rate: float) -> np.ndarray:
        if uniform_filter is None:
            return grid
        blurred = uniform_filter(grid, size=3, mode="reflect")
        return grid + rate * (blurred - grid)

    # ──────────────────────────────────────────────────────────
    # Per-epoch environmental update
    # ──────────────────────────────────────────────────────────
    def step(self, epoch: int) -> None:
        self.growth_modifier = (
            self.temperature_growth_factor(self.temperature)
            * self.ph_growth_factor(self.ph)
            * self.pressure_growth_factor(self.pressure_atm)
        )

        if self._use_gpu and self._tb is not None:
            self._step_gpu(epoch)
        else:
            self._step_cpu(epoch)

    # ── CPU path (original) ──
    def _step_cpu(self, epoch: int) -> None:
        self._update_resources_cpu()
        self._update_antibiotic_cpu(epoch)
        self._update_signal_cpu()
        self._update_biofilm_cpu()
        self._update_toxins_cpu()

    def _update_resources_cpu(self) -> None:
        self.resource = self._diffuse_cpu(self.resource, self.resource_diffusion)
        if self.resource_scenario == "resource-rich":
            self.resource += self.resource_replenishment
        np.clip(self.resource, 0.0, self.resource_max, out=self.resource)

    def _update_antibiotic_cpu(self, epoch: int) -> None:
        self._inject_antibiotic(epoch)
        self.antibiotic *= (1.0 - self.ab_decay)
        self.antibiotic = self._diffuse_cpu(self.antibiotic, self.ab_diffusion)
        ab_max = self.cfg.get("antibiotic", {}).get("max_concentration", 10.0)
        np.clip(self.antibiotic, 0.0, ab_max, out=self.antibiotic)

    def _update_signal_cpu(self) -> None:
        self.signal *= (1.0 - self.signal_decay)
        self.signal = self._diffuse_cpu(self.signal, self.signal_diffusion)
        np.clip(self.signal, 0.0, None, out=self.signal)

    def _update_biofilm_cpu(self) -> None:
        self.biofilm *= (1.0 - self.biofilm_decay)
        self.biofilm = self._diffuse_cpu(self.biofilm, 0.02)
        np.clip(self.biofilm, 0.0, None, out=self.biofilm)

    def _update_toxins_cpu(self) -> None:
        for g in list(self.toxin_grids):
            grid = self.toxin_grids[g]
            grid *= (1.0 - self.toxin_decay)
            self.toxin_grids[g] = self._diffuse_cpu(grid, self.toxin_diffusion)
            np.clip(self.toxin_grids[g], 0.0, None, out=self.toxin_grids[g])

    # ── GPU path: upload → compute → download ──
    def _step_gpu(self, epoch: int) -> None:
        tb = self._tb
        assert tb is not None

        # Upload numpy grids to GPU tensors
        t_res = tb.from_numpy(self.resource)
        t_ab = tb.from_numpy(self.antibiotic)
        t_sig = tb.from_numpy(self.signal)
        t_bio = tb.from_numpy(self.biofilm)

        # Resource: diffuse → replenish → clip
        t_res = tb.diffuse(t_res, self.resource_diffusion)
        if self.resource_scenario == "resource-rich":
            tb.add_scalar(t_res, self.resource_replenishment)
        tb.clip(t_res, 0.0, self.resource_max)

        # Antibiotic: inject on CPU (mode-dependent numpy logic), upload, then decay+diffuse+clip
        self._inject_antibiotic(epoch)
        t_ab = tb.from_numpy(self.antibiotic)  # re-upload after injection
        tb.decay(t_ab, self.ab_decay)
        t_ab = tb.diffuse(t_ab, self.ab_diffusion)
        ab_max = self.cfg.get("antibiotic", {}).get("max_concentration", 10.0)
        tb.clip(t_ab, 0.0, ab_max)

        # Signal: decay → diffuse → clip
        tb.decay(t_sig, self.signal_decay)
        t_sig = tb.diffuse(t_sig, self.signal_diffusion)
        tb.clip(t_sig, 0.0, None)

        # Biofilm: decay → diffuse → clip
        tb.decay(t_bio, self.biofilm_decay)
        t_bio = tb.diffuse(t_bio, 0.02)
        tb.clip(t_bio, 0.0, None)

        # Download back to numpy
        self.resource = self._from_tensor(t_res)
        self.antibiotic = self._from_tensor(t_ab)
        self.signal = self._from_tensor(t_sig)
        self.biofilm = self._from_tensor(t_bio)

        # Toxin grids — batch on GPU
        for g in list(self.toxin_grids):
            t_tox = tb.from_numpy(self.toxin_grids[g])
            tb.decay(t_tox, self.toxin_decay)
            t_tox = tb.diffuse(t_tox, self.toxin_diffusion)
            tb.clip(t_tox, 0.0, None)
            self.toxin_grids[g] = self._from_tensor(t_tox)

    # ──────────────────────────────────────────────────────────
    # Antibiotic injection (mode-dependent, shared by CPU & GPU)
    # ──────────────────────────────────────────────────────────
    def _inject_antibiotic(self, epoch: int) -> None:
        if epoch < self.ab_start_epoch:
            return
        elapsed = epoch - self.ab_start_epoch

        if self.ab_mode == "gradual":
            front_row = min(elapsed, self.height - 1)
            band = max(1, int(self.height * 0.05))
            for r in range(max(0, front_row - band), front_row + 1):
                self.antibiotic[r, :] += self.ab_gradual_rate
        elif self.ab_mode == "spike":
            if epoch == self.ab_start_epoch:
                self.antibiotic += self.ab_spike_conc
        elif self.ab_mode == "center":
            cy, cx = self.height / 2.0, self.width / 2.0
            radius = min(elapsed * 1.0, max(self.height, self.width) / 2.0)
            yy, xx = np.ogrid[:self.height, :self.width]
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            mask = dist <= radius
            falloff = np.where(mask, 1.0 - dist / (radius + 1e-9), 0.0)
            self.antibiotic += self.ab_gradual_rate * falloff
        elif self.ab_mode == "sweep":
            row = min(elapsed, self.height - 1)
            self.antibiotic[row, :] += self.ab_gradual_rate * 2.0

    # ──────────────────────────────────────────────────────────
    # Agent interaction helpers
    # ──────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────
    # Physics growth-factor models
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def temperature_growth_factor(
        T: float, T_min: float = 10.0, T_opt: float = 37.0, T_max: float = 45.0
    ) -> float:
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
        if pH <= pH_min or pH >= pH_max:
            return 0.0
        denom = (pH - pH_min) * (pH - pH_max) - (pH - pH_opt) ** 2
        if abs(denom) < 1e-12:
            return 0.0
        return max(0.0, min(1.0, ((pH - pH_min) * (pH - pH_max)) / denom))

    @staticmethod
    def pressure_growth_factor(P_atm: float) -> float:
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


