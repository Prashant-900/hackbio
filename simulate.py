"""
simulate.py — Simulation engine: epoch loop, agent orchestration, metric collection.

CSV output columns (PS-mandated):
  time_step, total_population, resource_concentration,
  genotype_0_density, genotype_1_density, …  (relative frequencies),
  mutation_frequency, cooperation_index, competition_index,
  mean_fitness, mean_resistance, mean_efficiency,
  biofilm_fraction, hgt_events, phase_lag, phase_log, phase_stationary, phase_death
"""

from __future__ import annotations

import csv
import os
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from agent import Bacterium, Genotype, Phase, reset_id_counter
from environment import Environment

try:
    from rl_agent import (
        BacterialAction,
        BacterialDQN,
        compute_reward,
        extract_states_batch,
    )

    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False


class Simulation:
    """Main simulation driver."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.env = Environment(cfg)
        self.agents: list[Bacterium] = []
        self.epoch: int = 0
        self.metrics: list[dict[str, Any]] = []

        # Per-epoch counters
        self.divisions_this_epoch: int = 0
        self.mutations_this_epoch: int = 0
        self.hgt_events_this_epoch: int = 0
        self.deaths_this_epoch: int = 0

        # Cumulative counters (LTEE: mutation accumulation is linear & clock-like)
        self.cumulative_mutations: int = 0
        self.cumulative_hgt: int = 0

        # RL brain (optional)
        self.dqn: BacterialDQN | None = None
        rl_cfg = cfg.get("rl", {})
        if RL_AVAILABLE and rl_cfg.get("enabled", False):
            self.dqn = BacterialDQN(cfg)

        # Seed
        seed = cfg["simulation"].get("seed")
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        reset_id_counter()
        self._spawn_initial_population()

    # ──────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────
    def _spawn_initial_population(self) -> None:
        n = self.cfg["bacterium"]["initial_count"]
        n_geno = self.cfg["genotype"]["initial_types"]
        w, h = self.cfg["grid"]["width"], self.cfg["grid"]["height"]

        for i in range(n):
            g_id = i % n_geno
            gt = Genotype(
                id=g_id,
                nutrient_efficiency=1.0 + random.gauss(0, 0.05),
                antibiotic_resistance=random.uniform(0.0, 0.15),
                toxin_production=random.uniform(0.3, 0.7),
                public_good_production=random.uniform(0.3, 0.7),
            )
            b = Bacterium(
                x=random.randint(0, w - 1),
                y=random.randint(0, h - 1),
                z=random.uniform(0, self.cfg.get("grid", {}).get("z_levels", 10)) if self.cfg.get('_view_mode', '2d') == '3d' else 0.0,
                genotype=gt,
            )
            b.attach_config(self.cfg)
            self.agents.append(b)

    # ──────────────────────────────────────────────────────────
    # Spatial index
    # ──────────────────────────────────────────────────────────
    def _build_spatial_index(self) -> dict[tuple[int, int], list[Bacterium]]:
        idx: dict[tuple[int, int], list[Bacterium]] = defaultdict(list)
        for a in self.agents:
            if a.alive:
                idx[(a.x, a.y)].append(a)
        return idx

    def _get_neighbours(
        self, agent: Bacterium,
        spatial: dict[tuple[int, int], list[Bacterium]],
    ) -> list[Bacterium]:
        r = self.cfg["hgt"]["radius"]
        neighbours: list[Bacterium] = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx == 0 and dy == 0:
                    continue
                key = (agent.x + dx, agent.y + dy)
                for nb in spatial.get(key, []):
                    if nb.uid != agent.uid and nb.alive:
                        neighbours.append(nb)
        return neighbours

    # ──────────────────────────────────────────────────────────
    # Single epoch
    # ──────────────────────────────────────────────────────────
    def step(self) -> None:
        self.epoch += 1
        self.divisions_this_epoch = 0
        self.mutations_this_epoch = 0
        self.hgt_events_this_epoch = 0
        self.deaths_this_epoch = 0

        # 1. Environment dynamics
        self.env.step(self.epoch)

        # 2. Agent actions
        new_agents: list[Bacterium] = []
        alive = [a for a in self.agents if a.alive]
        pop_size = len(alive)
        pre_alive = pop_size

        # RL: batch state extraction & action selection
        prev_biomasses: dict[int, float] = {}
        prev_states: dict[int, np.ndarray] = {}
        rl_actions: np.ndarray | None = None
        if self.dqn and self.dqn.enabled and alive:
            states = extract_states_batch(alive, self.env, self.cfg)
            rl_actions = self.dqn.select_actions_batch(states)
            for idx, agent in enumerate(alive):
                prev_biomasses[agent.uid] = agent.biomass
                prev_states[agent.uid] = states[idx]
                act = int(rl_actions[idx])
                agent._rl_action = act
                # Set RL flags based on action
                if act == int(BacterialAction.COOPERATE):
                    agent._rl_cooperate = True
                elif act == int(BacterialAction.COMPETE):
                    agent._rl_compete = True

        for agent in self.agents:
            if not agent.alive:
                continue
            # RL: movement bias from action
            enable_3d = getattr(self, 'view_mode', '2d') == '3d'
            if self.dqn and self.dqn.enabled:
                if agent._rl_action == int(BacterialAction.CHEMOTAXIS_RUN):
                    agent.move(self.env, run_bias=0.9, enable_3d=enable_3d)
                elif agent._rl_action == int(BacterialAction.CHEMOTAXIS_TUMBLE):
                    agent.move(self.env, run_bias=0.1, enable_3d=enable_3d)
                else:
                    agent.move(self.env, enable_3d=enable_3d)
            else:
                agent.move(self.env, enable_3d=enable_3d)
            daughter = agent.step(self.env, pop_size)
            if daughter is not None:
                new_agents.append(daughter)
                self.divisions_this_epoch += 1
                if daughter.mutated_this_division:
                    self.mutations_this_epoch += 1
            if not agent.alive:
                self.deaths_this_epoch += 1

        self.agents.extend(new_agents)

        # RL: compute rewards and store transitions
        if self.dqn and self.dqn.enabled and prev_states:
            # Build uid→agent lookup for O(1) access
            uid_to_agent = {a.uid: a for a in self.agents}
            # Track which parents produced daughters
            parent_uids = {d.uid for d in new_agents} if new_agents else set()
            # Build uid→action map from the pre-step rl_actions
            uid_to_action: dict[int, int] = {}
            for idx, a in enumerate(alive):
                if rl_actions is not None:
                    uid_to_action[a.uid] = int(rl_actions[idx])

            for uid, prev_state in prev_states.items():
                bact = uid_to_agent.get(uid)
                act = uid_to_action.get(uid, 6)
                if bact is None:
                    # fully removed (shouldn't happen before dead cleanup)
                    self.dqn.store(
                        prev_state, act, -10.0,
                        np.zeros(14, dtype=np.float32), True,
                    )
                    continue
                divided = any(
                    d for d in new_agents
                    if d.x in range(max(0, bact.x - 1), bact.x + 2)
                    and d.y in range(max(0, bact.y - 1), bact.y + 2)
                ) if new_agents else False
                reward = compute_reward(
                    bact, prev_biomasses.get(uid, bact.biomass),
                    divided, bact.alive,
                )
                if bact.alive:
                    from rl_agent import extract_state
                    next_state = extract_state(bact, self.env, self.cfg)
                else:
                    next_state = np.zeros(14, dtype=np.float32)
                self.dqn.store(
                    prev_state, act,
                    reward, next_state, not bact.alive,
                )

            # Train DQN periodically
            if self.epoch % self.dqn.train_every == 0:
                self.dqn.train_step()

        # 3. HGT (every 5 epochs, sampled for performance)
        if self.epoch % 5 == 0:
            spatial = self._build_spatial_index()
            sample = (self.agents if len(self.agents) < 500
                      else random.sample(self.agents, 500))
            for agent in sample:
                if not agent.alive:
                    continue
                nbs = self._get_neighbours(agent, spatial)
                if agent.attempt_hgt(nbs, self.cfg):
                    self.hgt_events_this_epoch += 1

        # Accumulate into cumulative counters (LTEE clock-like tracking)
        self.cumulative_mutations += self.mutations_this_epoch
        self.cumulative_hgt += self.hgt_events_this_epoch

        # 4. Remove dead
        self.agents = [a for a in self.agents if a.alive]

        # 5. Cleanup toxin grids every 20 epochs
        if self.epoch % 20 == 0:
            active = {a.genotype.id for a in self.agents}
            self.env.cleanup_toxin_grids(active)

        # 6. Record metrics
        self._record_metrics()

    # ──────────────────────────────────────────────────────────
    # Metrics (PS-compliant)
    # ──────────────────────────────────────────────────────────
    def _record_metrics(self) -> None:
        alive = [a for a in self.agents if a.alive]
        total_pop = len(alive)

        # Genotype counts & relative densities
        geno_counts: Counter[int] = Counter(a.genotype.id for a in alive)
        geno_density: dict[int, float] = {}
        for g, cnt in geno_counts.items():
            geno_density[g] = round(cnt / total_pop, 6) if total_pop > 0 else 0.0

        # Phase distribution
        phase_counts = Counter(a.phase.name for a in alive)

        # Cooperation index: fraction in biofilm × mean public_good_production
        n_biofilm = sum(1 for a in alive if a.biofilm_member)
        biofilm_frac = n_biofilm / total_pop if total_pop > 0 else 0.0
        mean_pg = (np.mean([a.genotype.public_good_production for a in alive])
                   if total_pop > 0 else 0.0)
        coop_index = round(biofilm_frac * float(mean_pg), 6)

        # Competition index: mean foreign toxin experienced normalized
        if total_pop > 0:
            comp_vals = [self.env.get_foreign_toxin(a.genotype.id, a.x, a.y) for a in alive]
            comp_index = round(float(np.mean(comp_vals)), 6)
        else:
            comp_index = 0.0

        # Mutation frequency: mutations / divisions this epoch
        mut_freq = (self.mutations_this_epoch / self.divisions_this_epoch
                    if self.divisions_this_epoch > 0 else 0.0)

        # Mean fitness, resistance, efficiency
        if total_pop > 0:
            mean_fit = round(float(np.mean([a.fitness for a in alive])), 6)
            mean_res = round(float(np.mean([a.genotype.antibiotic_resistance for a in alive])), 6)
            mean_eff = round(float(np.mean([a.genotype.nutrient_efficiency for a in alive])), 6)
        else:
            mean_fit = mean_res = mean_eff = 0.0

        row: dict[str, Any] = {
            "time_step": self.epoch,
            "total_population": total_pop,
            "resource_concentration": round(self.env.mean_resource(), 6),
            "genotype_counts": dict(geno_counts),
            "genotype_density": geno_density,
            "mutation_frequency": round(mut_freq, 6),
            "cooperation_index": round(coop_index, 6),
            "competition_index": comp_index,
            "mean_fitness": mean_fit,
            "mean_resistance": mean_res,
            "mean_efficiency": mean_eff,
            "biofilm_fraction": round(biofilm_frac, 6),
            "hgt_events": self.hgt_events_this_epoch,
            "divisions": self.divisions_this_epoch,
            "deaths": self.deaths_this_epoch,
            "phase_lag": phase_counts.get("LAG", 0),
            "phase_log": phase_counts.get("LOG", 0),
            "phase_stationary": phase_counts.get("STATIONARY", 0),
            "phase_death": phase_counts.get("DEATH", 0),
            "mean_antibiotic": round(self.env.mean_antibiotic(), 6),
            "mean_biofilm": round(self.env.mean_biofilm(), 6),
            "cumulative_mutations": self.cumulative_mutations,
            "cumulative_hgt": self.cumulative_hgt,
            "total_resource_consumed": round(self.env.total_resource_consumed, 4),
        }
        self.metrics.append(row)

    # ──────────────────────────────────────────────────────────
    # Run loop
    # ──────────────────────────────────────────────────────────
    def run(self, callback=None) -> list[dict[str, Any]]:
        total_epochs = self.cfg["simulation"]["epochs"]
        for _ in range(total_epochs):
            self.step()
            if callback:
                callback(self.epoch, self)
            if len(self.agents) == 0:
                print(f"[epoch {self.epoch}] Population extinct — stopping.")
                break
        return self.metrics

    # ──────────────────────────────────────────────────────────
    # CSV export (PS-compliant columns)
    # ──────────────────────────────────────────────────────────
    def export_csv(self, path: str | None = None) -> str:
        out_dir = self.cfg["simulation"]["output_dir"]
        os.makedirs(out_dir, exist_ok=True)
        if path is None:
            path = os.path.join(out_dir, self.cfg["simulation"]["csv_filename"])

        # Collect all genotype IDs that ever appeared
        all_genos: set[int] = set()
        for m in self.metrics:
            all_genos.update(m["genotype_density"].keys())
        sorted_genos = sorted(all_genos)

        fieldnames = [
            "time_step", "total_population", "resource_concentration",
        ]
        # Add per-genotype density columns
        for g in sorted_genos:
            fieldnames.append(f"genotype_{g}_density")
        fieldnames += [
            "mutation_frequency", "cooperation_index", "competition_index",
            "mean_fitness", "mean_resistance", "mean_efficiency",
            "biofilm_fraction", "hgt_events", "divisions", "deaths",
            "phase_lag", "phase_log", "phase_stationary", "phase_death",
            "mean_antibiotic", "mean_biofilm",
            "cumulative_mutations", "cumulative_hgt", "total_resource_consumed",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in self.metrics:
                row: dict[str, Any] = {
                    "time_step": m["time_step"],
                    "total_population": m["total_population"],
                    "resource_concentration": m["resource_concentration"],
                }
                for g in sorted_genos:
                    row[f"genotype_{g}_density"] = m["genotype_density"].get(g, 0.0)
                row.update({
                    "mutation_frequency": m["mutation_frequency"],
                    "cooperation_index": m["cooperation_index"],
                    "competition_index": m["competition_index"],
                    "mean_fitness": m["mean_fitness"],
                    "mean_resistance": m["mean_resistance"],
                    "mean_efficiency": m["mean_efficiency"],
                    "biofilm_fraction": m["biofilm_fraction"],
                    "hgt_events": m["hgt_events"],
                    "divisions": m["divisions"],
                    "deaths": m["deaths"],
                    "phase_lag": m["phase_lag"],
                    "phase_log": m["phase_log"],
                    "phase_stationary": m["phase_stationary"],
                    "phase_death": m["phase_death"],
                    "mean_antibiotic": m["mean_antibiotic"],
                    "mean_biofilm": m["mean_biofilm"],
                    "cumulative_mutations": m["cumulative_mutations"],
                    "cumulative_hgt": m["cumulative_hgt"],
                    "total_resource_consumed": m["total_resource_consumed"],
                })
                writer.writerow(row)
        return path
