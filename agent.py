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
    age: int = 0
    biomass: float = 0.5
    phase: Phase = Phase.LAG
    genotype: Genotype = field(default_factory=lambda: Genotype(0, 1.0, 0.0, 0.5, 0.5))
    alive: bool = True
    biofilm_member: bool = False
    fitness: float = 0.0          # computed each epoch
    mutated_this_division: bool = False  # track for mutation_frequency metric

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
        daughter = None
        if self.phase in (Phase.LOG, Phase.STATIONARY):
            growth = self._monod_growth(local_res)
            if self.phase == Phase.STATIONARY:
                growth *= 0.1

            yld = self._cfg["monod"]["yield_coefficient"]
            consumed = env.consume_resource(self.x, self.y, growth * yld)
            self.biomass += consumed / yld if yld > 0 else 0.0

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
            eps_amount = self.genotype.public_good_production * 0.1
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
        tox_amount = tox_cfg["secretion_rate"] * self.genotype.toxin_production
        tox_cost = tox_amount * self._cfg["bacterium"]["toxin_production_cost"]
        if self.biomass > tox_cost and self.phase in (Phase.LOG, Phase.STATIONARY):
            self.biomass -= tox_cost
            env.add_toxin(self.genotype.id, self.x, self.y, tox_amount)

        # ── Death check ──
        if self._death_check(env):
            self.alive = False

        return daughter

    # ──────────────────────────────────────────────────────────
    # Binary fission & mutation
    # ──────────────────────────────────────────────────────────
    def _divide(self, env: "Environment") -> "Bacterium":
        self.biomass *= 0.5

        child_genotype = self.genotype.copy()
        mut_cfg = self._cfg["mutation"]
        mutated = False
        if random.random() < mut_cfg["rate"]:
            child_genotype = self._mutate(child_genotype, mut_cfg)
            mutated = True

        # Daughter placed in random adjacent cell
        dx, dy = random.choice([
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ])
        nx = max(0, min(env.width - 1, self.x + dx))
        ny = max(0, min(env.height - 1, self.y + dy))

        daughter = Bacterium(
            x=nx, y=ny, age=0, biomass=self.biomass,
            phase=Phase.LAG, genotype=child_genotype,
        )
        daughter.mutated_this_division = mutated
        daughter.attach_config(self._cfg)
        env.ensure_genotype_toxin_grid(child_genotype.id)
        return daughter

    @staticmethod
    def _mutate(gt: Genotype, mut_cfg: dict) -> Genotype:
        """Point mutations — small perturbations to phenotypic traits."""
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

        # Novel genotype lineage (speciation event)
        max_types = 20
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
        ab_death = ab * (1.0 - resistance) * 0.1

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
        """Transfer resistance or toxin trait from a neighbour. Returns True if HGT occurred."""
        hgt_cfg = cfg["hgt"]
        for nb in neighbours:
            if nb.genotype.id == self.genotype.id:
                continue
            if random.random() < hgt_cfg["probability"]:
                # Swap antibiotic resistance (plasmid transfer)
                self.genotype.antibiotic_resistance, nb.genotype.antibiotic_resistance = (
                    nb.genotype.antibiotic_resistance,
                    self.genotype.antibiotic_resistance,
                )
                # Optionally swap toxin production trait
                if random.random() < 0.3:
                    self.genotype.toxin_production, nb.genotype.toxin_production = (
                        nb.genotype.toxin_production,
                        self.genotype.toxin_production,
                    )
                return True
        return False

    # ──────────────────────────────────────────────────────────
    # Chemotaxis-inspired movement
    # ──────────────────────────────────────────────────────────
    def move(self, env: "Environment") -> None:
        """Biased random walk toward higher resource (chemotaxis)."""
        if self.phase not in (Phase.LOG, Phase.STATIONARY):
            return

        # Evaluate resource in each neighbour direction
        best_dx, best_dy = 0, 0
        best_res = env.resource[self.y, self.x]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]

        if random.random() < 0.6:  # 60% biased, 40% random (run-and-tumble)
            for dx, dy in directions:
                nx = max(0, min(env.width - 1, self.x + dx))
                ny = max(0, min(env.height - 1, self.y + dy))
                r = env.resource[ny, nx]
                if r > best_res:
                    best_res = r
                    best_dx, best_dy = dx, dy
        else:
            best_dx, best_dy = random.choice(directions)

        self.x = max(0, min(env.width - 1, self.x + best_dx))
        self.y = max(0, min(env.height - 1, self.y + best_dy))
