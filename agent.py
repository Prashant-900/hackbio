"""
agent.py — Bacterium agent with biologically grounded mechanics.

Biological basis:
  - Growth: Monod kinetics (μ = μ_max · S / (K_s + S))
  - Life-cycle: Lag → Log → Stationary → Death (classical growth curve)
  - Metabolism: Maintenance energy cost (Herbert 1958, endogenous metabolism)
  - Reproduction: Binary fission when biomass ≥ threshold
  - Mutation: Stochastic trait perturbation during DNA replication
  - Fitness: Multi-trait weighted score (Fisher's geometric model)
  - Toxin: Bacteriocin secretion at metabolic cost (Riley & Wertz 2002)
  - Cooperation: EPS biofilm matrix as public good (cost to producer)
  - Quorum Sensing: AHL signal → collective biofilm activation (Fuqua 1994)
  - HGT: Conjugative transfer of resistance/toxin traits (Frost 2005)
  - Death: Stochastic — f(age, starvation, antibiotic, foreign toxin, density)

Conservation: All growth consumes resources. Toxin & EPS production cost biomass.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment import Environment

# ──────────────────────────────────────────────────────────────
# Unique ID generator
# ──────────────────────────────────────────────────────────────
_next_id: int = 0


def _new_id() -> int:
    global _next_id
    _next_id += 1
    return _next_id


def reset_id_counter() -> None:
    global _next_id
    _next_id = 0


# ──────────────────────────────────────────────────────────────
# Growth phase enum
# ──────────────────────────────────────────────────────────────
class Phase(Enum):
    LAG = auto()
    LOG = auto()
    STATIONARY = auto()
    DEATH = auto()


# ──────────────────────────────────────────────────────────────
# Genotype: heritable trait vector
# ──────────────────────────────────────────────────────────────
@dataclass
class Genotype:
    id: int                        # lineage label
    nutrient_efficiency: float     # multiplier on μ_max (substrate affinity)
    antibiotic_resistance: float   # 0 = susceptible, 1 = fully resistant
    toxin_production: float        # bacteriocin secretion intensity [0,1]
    public_good_production: float  # EPS/biofilm contribution [0,1]

    def copy(self) -> "Genotype":
        return Genotype(
            id=self.id,
            nutrient_efficiency=self.nutrient_efficiency,
            antibiotic_resistance=self.antibiotic_resistance,
            toxin_production=self.toxin_production,
            public_good_production=self.public_good_production,
        )


# ──────────────────────────────────────────────────────────────
# Bacterium agent
# ──────────────────────────────────────────────────────────────
@dataclass
class Bacterium:
    uid: int = field(default_factory=_new_id)
    x: int = 0
    y: int = 0
    z: float = 0.0                    # depth in 3D colony (0 = surface)
    age: int = 0
    biomass: float = 0.5
    phase: Phase = Phase.LAG
    genotype: Genotype = field(default_factory=lambda: Genotype(0, 1.0, 0.0, 0.5, 0.5))
    alive: bool = True
    biofilm_member: bool = False
    fitness: float = 0.0          # computed each epoch
    mutated_this_division: bool = False  # track for mutation_frequency metric

    # RL action flags (set externally by DQN before step)
    _rl_action: int = 6           # default = GROW
    _rl_cooperate: bool = False
    _rl_compete: bool = False

    _cfg: dict = field(default_factory=dict, repr=False)
    _carrying_ratio: float = field(default=0.5, repr=False)

    def attach_config(self, cfg: dict) -> None:
        self._cfg = cfg

    # ──────────────────────────────────────────────────────────
    # Monod growth rate
    # ──────────────────────────────────────────────────────────
    def _monod_growth(self, resource: float) -> float:
        m = self._cfg["monod"]
        mu_max = m["mu_max"] * self.genotype.nutrient_efficiency
        Ks = m["Ks"]
        return mu_max * resource / (Ks + resource) if (Ks + resource) > 0 else 0.0

    # ──────────────────────────────────────────────────────────
    # Fitness score (multi-trait weighted landscape)
    # ──────────────────────────────────────────────────────────
    def compute_fitness(self, local_resource: float, local_antibiotic: float) -> float:
        f_cfg = self._cfg["fitness"]
        growth = self._monod_growth(local_resource)
        norm_growth = min(1.0, growth / max(self._cfg["monod"]["mu_max"], 0.01))
        resist = self.genotype.antibiotic_resistance
        effic = min(1.5, self.genotype.nutrient_efficiency) / 1.5
        coop = self.genotype.public_good_production

        # Antibiotic penalty if antibiotic present and not resistant
        ab_penalty = local_antibiotic * (1.0 - resist) * 0.1

        self.fitness = max(0.0,
            f_cfg["weight_growth"] * norm_growth
            + f_cfg["weight_resistance"] * resist
            + f_cfg["weight_efficiency"] * effic
            + f_cfg["weight_cooperation"] * coop
            - ab_penalty
        )
        return self.fitness

    # ──────────────────────────────────────────────────────────
    # Phase transitions (classical bacterial growth curve)
    # ──────────────────────────────────────────────────────────
    def _update_phase(self, local_resource: float, carrying_ratio: float) -> None:
        lag_dur = self._cfg["bacterium"]["lag_phase_duration"]

        if self.phase == Phase.LAG:
            if self.age >= lag_dur:
                self.phase = Phase.LOG
        elif self.phase == Phase.LOG:
            if carrying_ratio >= 1.0 or local_resource < 0.05:
                self.phase = Phase.STATIONARY
        elif self.phase == Phase.STATIONARY:
            if local_resource < 0.01:
                self.phase = Phase.DEATH
            elif carrying_ratio < 0.8 and local_resource > 0.5:
                self.phase = Phase.LOG
        # DEATH phase handled in death check

    # ──────────────────────────────────────────────────────────
    # Main per-epoch step
    # ──────────────────────────────────────────────────────────
    def step(self, env: "Environment", population_size: int) -> "Bacterium | None":
        if not self.alive:
            return None

        self.age += 1
        self.mutated_this_division = False
        carrying = self._cfg["population"]["carrying_capacity"]
        self._carrying_ratio = population_size / carrying if carrying > 0 else 1.0
        local_res = env.resource[self.y, self.x]
        local_ab = env.antibiotic[self.y, self.x]

        # Phase transition
        self._update_phase(local_res, self._carrying_ratio)

        # Compute fitness
        self.compute_fitness(local_res, local_ab)

        # ── Maintenance metabolism (conservation law) ──
        maint = self._cfg["bacterium"]["maintenance_energy"]
        self.biomass -= maint
        if self.biomass < 0:
            self.biomass = 0.0

        # ── Growth (LOG / STATIONARY) ──
        # Monod kinetics + mass-balance:
        #   μ  = μ_max · S / (K_s + S)          … specific growth rate
        #   ΔX = μ                                … biomass gain per cell
        #   ΔS = μ / Y_{X/S}                     … substrate required
        # If resource is insufficient, actual ΔX = consumed · Y.
        # Reference: Monod (1949), Herbert (1958)
        daughter = None
        if self.phase in (Phase.LOG, Phase.STATIONARY):
            growth = self._monod_growth(local_res)
            if self.phase == Phase.STATIONARY:
                growth *= 0.1

            # Physics: environment growth modifier (temperature × pH × pressure)
            growth *= env.growth_modifier

            # 3D depth modifier — deeper bacteria have reduced nutrient access
            z_depth = self.z / max(getattr(env, 'z_levels', 10), 1)
            growth *= (1.0 - 0.3 * z_depth)

            # RL: CONSERVE action halves growth
            if self._rl_action == 5:  # CONSERVE
                growth *= 0.5

            yld = self._cfg["monod"]["yield_coefficient"]  # Y_{X/S}
            substrate_needed = growth / yld if yld > 0 else growth
            consumed = env.consume_resource(self.x, self.y, substrate_needed)
            self.biomass += consumed * yld

            # Division (suppressed above carrying capacity)
            if self.biomass >= self._cfg["bacterium"]["division_threshold"]:
                if self._carrying_ratio < 1.0:
                    daughter = self._divide(env)
                else:
                    self.biomass = self._cfg["bacterium"]["division_threshold"] * 0.9

        # ── Quorum sensing: produce AHL signal ──
        qs_cfg = self._cfg["quorum_sensing"]
        env.add_signal(self.x, self.y, qs_cfg["signal_production_rate"])
        local_signal = env.signal[self.y, self.x]
        self.biofilm_member = local_signal >= qs_cfg["activation_threshold"]

        # ── Cooperation: EPS biofilm production (at metabolic cost) ──
        if self.biofilm_member and self.phase in (Phase.LOG, Phase.STATIONARY):
            eps_mult = 2.0 if self._rl_cooperate else 1.0
            eps_amount = self.genotype.public_good_production * 0.1 * eps_mult
            cost = eps_amount * self._cfg["bacterium"]["public_good_cost"]
            if self.biomass > cost:
                self.biomass -= cost
                env.add_biofilm(self.x, self.y, eps_amount)

            # Resource sharing within biofilm (public good)
            share_frac = qs_cfg.get("biofilm_resource_sharing", 0.0)
            if share_frac > 0 and local_res > 0.1:
                shared = env.consume_resource(self.x, self.y, share_frac * local_res)
                # Redistribute to neighbours
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx = max(0, min(env.width - 1, self.x + dx))
                        ny = max(0, min(env.height - 1, self.y + dy))
                        env.resource[ny, nx] += shared / 8.0

        # ── Competition: bacteriocin secretion (at metabolic cost) ──
        tox_cfg = self._cfg["toxin"]
        tox_mult = 2.0 if self._rl_compete else 1.0
        tox_amount = tox_cfg["secretion_rate"] * self.genotype.toxin_production * tox_mult
        tox_cost = tox_amount * self._cfg["bacterium"]["toxin_production_cost"]
        if self.biomass > tox_cost and self.phase in (Phase.LOG, Phase.STATIONARY):
            self.biomass -= tox_cost
            env.add_toxin(self.genotype.id, self.x, self.y, tox_amount)

        # ── Death check ──
        if self._death_check(env):
            self.alive = False

        # Reset RL flags for next epoch
        self._rl_action = 6
        self._rl_cooperate = False
        self._rl_compete = False

        return daughter

    # ──────────────────────────────────────────────────────────
    # Binary fission & mutation
    # ──────────────────────────────────────────────────────────
    def _divide(self, env: "Environment") -> "Bacterium":
        self.biomass *= 0.5

        child_genotype = self.genotype.copy()
        mut_cfg = self._cfg["mutation"]
        max_types = self._cfg["genotype"]["max_types"]
        mutated = False
        if random.random() < mut_cfg["rate"]:
            child_genotype = self._mutate(child_genotype, mut_cfg, max_types)
            mutated = True

        # Daughter placed in random adjacent cell
        dx, dy = random.choice([
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ])
        nx = max(0, min(env.width - 1, self.x + dx))
        ny = max(0, min(env.height - 1, self.y + dy))

        # In 2D mode z stays 0; in 3D mode daughter drifts in z
        is_3d = self._cfg.get('_view_mode', '2d') == '3d'
        child_z = self.z + random.uniform(-0.5, 0.5) if is_3d else 0.0
        daughter = Bacterium(
            x=nx, y=ny, z=child_z,
            age=0, biomass=self.biomass,
            phase=Phase.LAG, genotype=child_genotype,
        )
        if is_3d:
            daughter.z = max(0.0, min(float(self._cfg.get('grid', {}).get('z_levels', 10)), daughter.z))
        daughter.mutated_this_division = mutated
        daughter.attach_config(self._cfg)
        env.ensure_genotype_toxin_grid(child_genotype.id)
        return daughter

    @staticmethod
    def _mutate(gt: Genotype, mut_cfg: dict, max_types: int = 20) -> Genotype:
        """Point mutations — small perturbations to phenotypic traits.

        Models imperfect DNA replication during binary fission.
        Trait perturbations are drawn from uniform distributions
        bounded by per-trait delta values — analogous to the
        distribution of fitness effects (DFE) of new mutations.
        A 10 % chance of lineage reassignment models rare
        large-effect mutations that found novel clades (LTEE
        shows ~10–20 beneficial fixations per 20 000 generations).
        """
        gt.nutrient_efficiency += random.uniform(
            -mut_cfg["efficiency_delta"], mut_cfg["efficiency_delta"]
        )
        gt.nutrient_efficiency = max(0.1, gt.nutrient_efficiency)

        gt.antibiotic_resistance += random.uniform(
            -mut_cfg["resistance_delta"], mut_cfg["resistance_delta"]
        )
        gt.antibiotic_resistance = max(0.0, min(1.0, gt.antibiotic_resistance))

        gt.toxin_production += random.uniform(
            -mut_cfg.get("toxin_production_delta", 0.03),
            mut_cfg.get("toxin_production_delta", 0.03),
        )
        gt.toxin_production = max(0.0, min(1.0, gt.toxin_production))

        gt.public_good_production += random.uniform(
            -mut_cfg.get("public_good_delta", 0.03),
            mut_cfg.get("public_good_delta", 0.03),
        )
        gt.public_good_production = max(0.0, min(1.0, gt.public_good_production))

        # Novel genotype lineage — rare large-effect mutation
        if random.random() < 0.1:
            gt.id = random.randint(0, max_types - 1)
        return gt

    # ──────────────────────────────────────────────────────────
    # Death probability
    # ──────────────────────────────────────────────────────────
    def _death_check(self, env: "Environment") -> bool:
        cfg_b = self._cfg["bacterium"]
        base = cfg_b["base_death_rate"]
        max_age = cfg_b["max_age"]

        # Age-dependent senescence
        age_factor = (self.age / max_age) ** 2 if max_age > 0 else 0.0

        # Starvation
        local_res = env.resource[self.y, self.x]
        starvation = 0.12 if local_res < 0.01 else (0.03 if local_res < 0.1 else 0.0)

        # Antibiotic (reduced by resistance + biofilm EPS)
        ab = env.antibiotic[self.y, self.x]
        resistance = self.genotype.antibiotic_resistance
        if self.biofilm_member:
            biofilm_shield = env.biofilm[self.y, self.x]
            ab *= self._cfg["quorum_sensing"]["biofilm_resistance_multiplier"]
            ab *= max(0.3, 1.0 - biofilm_shield * 0.1)  # EPS reduces penetration
        # Saturating Hill-function death: prevents runaway lethality at high [AB]
        effective_ab = ab * (1.0 - resistance)
        ab_death = 0.2 * effective_ab / (effective_ab + 2.0)  # half-max at 2.0

        # Foreign toxin (competitive exclusion)
        foreign_tox = env.get_foreign_toxin(self.genotype.id, self.x, self.y)
        tox_death = foreign_tox * self._cfg["toxin"]["lethality"]

        # DEATH-phase acceleration
        phase_boost = 0.15 if self.phase == Phase.DEATH else 0.0

        # Density-dependent (logistic) death pressure
        density_pressure = max(0.0, (self._carrying_ratio - 0.8)) * 0.3

        # Biomass depletion
        biomass_death = 0.2 if self.biomass <= 0 else 0.0

        total = (base + age_factor + starvation + ab_death
                 + tox_death + phase_boost + density_pressure + biomass_death)
        return random.random() < min(total, 0.95)

    # ──────────────────────────────────────────────────────────
    # Horizontal Gene Transfer (conjugation)
    # ──────────────────────────────────────────────────────────
    def attempt_hgt(self, neighbours: list["Bacterium"], cfg: dict) -> bool:
        """Conjugative plasmid transfer from a neighbour donor (one-way).

        Biological basis (Frost et al. 2005):
          - Donor transfers a *copy* of its plasmid to the recipient.
          - Donor retains its own traits (no loss).
          - Recipient acquires the donor's resistance (takes max if already
            partially resistant, modelling additive plasmid acquisition).
          - Optionally the donor's toxin-production cassette is co-transferred
            (linked genes on the same plasmid, ~30 % probability).
        """
        hgt_cfg = cfg["hgt"]
        for nb in neighbours:
            if nb.genotype.id == self.genotype.id:
                continue
            if random.random() < hgt_cfg["probability"]:
                # Recipient (self) acquires donor (nb) resistance — one-way
                self.genotype.antibiotic_resistance = max(
                    self.genotype.antibiotic_resistance,
                    nb.genotype.antibiotic_resistance,
                )
                # Co-transfer of toxin-production cassette (~30 %)
                if random.random() < 0.3:
                    self.genotype.toxin_production = max(
                        self.genotype.toxin_production,
                        nb.genotype.toxin_production,
                    )
                return True
        return False

    # ──────────────────────────────────────────────────────────
    # Chemotaxis-inspired movement
    # ──────────────────────────────────────────────────────────
    def move(self, env: "Environment", run_bias: float = 0.6, enable_3d: bool = False) -> None:
        """Biased random walk toward higher resource (chemotaxis).

        Models the *run-and-tumble* motility of E. coli:
          - run_bias fraction of moves are *biased* (run toward the
            steepest nutrient gradient in the Moore neighbourhood).
          - (1 - run_bias) moves are random *tumbles*.
        Includes all 8 neighbour directions (Moore neighbourhood)
        plus staying in place.

        When enable_3d is True, bacteria also move along the z-axis.
        """
        if self.phase not in (Phase.LOG, Phase.STATIONARY):
            return

        best_dx, best_dy = 0, 0
        best_res = env.resource[self.y, self.x]
        # Moore neighbourhood (8 directions + stay)
        directions = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
            (0, 0),
        ]

        if random.random() < run_bias:  # biased run
            for dx, dy in directions:
                nx = max(0, min(env.width - 1, self.x + dx))
                ny = max(0, min(env.height - 1, self.y + dy))
                r = env.resource[ny, nx]
                if r > best_res:
                    best_res = r
                    best_dx, best_dy = dx, dy
        else:  # random tumble
            best_dx, best_dy = random.choice(directions)

        self.x = max(0, min(env.width - 1, self.x + best_dx))
        self.y = max(0, min(env.height - 1, self.y + best_dy))

        # 3D z-axis movement when in 3D mode
        if enable_3d:
            z_levels = getattr(env, 'z_levels', self._cfg.get('grid', {}).get('z_levels', 10))
            dz = random.choice([-1, 0, 0, 0, 1])  # mostly stay, sometimes drift
            # Bias toward surface (z=0) where nutrients are richer
            if self.z > z_levels * 0.5 and random.random() < 0.3:
                dz = -1
            self.z = max(0.0, min(float(z_levels), self.z + dz))
