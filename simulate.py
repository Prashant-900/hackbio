"""
simulate.py — Simulation engine: epoch loop, agent orchestration, metric collection.

GPU acceleration:
  When a CUDA/MPS device is available the heaviest per-epoch work is batched
  into contiguous numpy/torch arrays and processed in bulk:

  1. Environment diffusion/decay/clip — runs entirely on GPU via TensorBackend
  2. Movement (chemotaxis) — vectorised best-neighbour lookup on GPU tensors
  3. Growth (Monod) — vectorised μ, substrate consumption, biomass gain
  4. Fitness — vectorised multi-trait weighted score
  5. Death — vectorised probability roll
  6. Toxin/signal/biofilm production — scatter-add on GPU grids

  Only division (binary fission + mutation) and HGT remain sequential because
  they create new agent objects with stochastic state.

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
    import torch
    from gpu_utils import TensorBackend, get_device
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

try:
    from rl_agent import (
        BacterialAction,
        BacterialDQN,
        compute_reward,
        extract_state,
        extract_states_batch,
    )

    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

# Phase int encoding for vectorised ops
_PHASE_INT = {Phase.LAG: 0, Phase.LOG: 1, Phase.STATIONARY: 2, Phase.DEATH: 3}
_INT_PHASE = {0: Phase.LAG, 1: Phase.LOG, 2: Phase.STATIONARY, 3: Phase.DEATH}


class Simulation:
    """Main simulation driver."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        force_cpu = cfg.get("rl", {}).get("force_cpu", False)
        self.env = Environment(cfg, force_cpu=force_cpu)
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

        # 1. Environment dynamics (GPU-accelerated when available)
        self.env.step(self.epoch)

        # 2. Agent actions — GPU batch path or CPU scalar path
        alive = [a for a in self.agents if a.alive]
        pop_size = len(alive)

        # RL: batch state extraction & action selection (always on GPU if RL enabled)
        prev_biomasses: dict[int, float] = {}
        prev_states: dict[int, np.ndarray] = {}
        rl_actions: np.ndarray | None = None
        rl_enabled = self.dqn and self.dqn.enabled
        if rl_enabled and alive:
            states = extract_states_batch(alive, self.env, self.cfg)
            rl_actions = self.dqn.select_actions_batch(states)
            for idx, agent in enumerate(alive):
                prev_biomasses[agent.uid] = agent.biomass
                prev_states[agent.uid] = states[idx]
                act = int(rl_actions[idx])
                agent._rl_action = act
                if act == int(BacterialAction.COOPERATE):
                    agent._rl_cooperate = True
                elif act == int(BacterialAction.COMPETE):
                    agent._rl_compete = True

        # Choose batch (GPU/vectorised) or scalar (CPU per-agent) path
        use_batch = (self.env._use_gpu or pop_size >= 200) and _TORCH_OK
        if use_batch:
            new_agents = self._batch_step(alive, pop_size, rl_enabled, rl_actions)
        else:
            new_agents = self._scalar_step(alive, pop_size, rl_enabled)

        self.agents.extend(new_agents)

        # RL: compute rewards and store transitions
        if rl_enabled and prev_states:
            uid_to_agent = {a.uid: a for a in self.agents}
            uid_to_action: dict[int, int] = {}
            for idx, a in enumerate(alive):
                if rl_actions is not None:
                    uid_to_action[a.uid] = int(rl_actions[idx])

            daughter_cells: set[tuple[int, int]] = set()
            for d in new_agents:
                daughter_cells.add((d.x, d.y))

            _zero_state = np.zeros(14, dtype=np.float32)
            for uid, prev_state in prev_states.items():
                bact = uid_to_agent.get(uid)
                act = uid_to_action.get(uid, 6)
                if bact is None:
                    self.dqn.store(prev_state, act, -10.0, _zero_state, True)
                    continue
                divided = False
                if daughter_cells:
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if (bact.x + dx, bact.y + dy) in daughter_cells:
                                divided = True
                                break
                        if divided:
                            break
                reward = compute_reward(
                    bact, prev_biomasses.get(uid, bact.biomass),
                    divided, bact.alive,
                )
                if bact.alive:
                    next_state = extract_state(bact, self.env, self.cfg)
                else:
                    next_state = _zero_state
                self.dqn.store(prev_state, act, reward, next_state, not bact.alive)

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

        # Accumulate into cumulative counters
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
    # Scalar (CPU) path — original per-agent loop
    # ──────────────────────────────────────────────────────────
    def _scalar_step(self, alive: list[Bacterium], pop_size: int,
                     rl_enabled: bool) -> list[Bacterium]:
        new_agents: list[Bacterium] = []
        enable_3d = getattr(self, 'view_mode', '2d') == '3d'
        _RUN = int(BacterialAction.CHEMOTAXIS_RUN) if RL_AVAILABLE else -1
        _TUMBLE = int(BacterialAction.CHEMOTAXIS_TUMBLE) if RL_AVAILABLE else -1

        for agent in self.agents:
            if not agent.alive:
                continue
            if rl_enabled:
                rl_act = agent._rl_action
                if rl_act == _RUN:
                    agent.move(self.env, run_bias=0.9, enable_3d=enable_3d)
                elif rl_act == _TUMBLE:
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
        return new_agents

    # ──────────────────────────────────────────────────────────
    # Batch (GPU-vectorised) path
    # ──────────────────────────────────────────────────────────
    def _batch_step(self, alive: list[Bacterium], pop_size: int,
                    rl_enabled: bool,
                    rl_actions: np.ndarray | None) -> list[Bacterium]:
        """Vectorised agent step: movement, Monod growth, fitness, death,
        toxin/signal/biofilm production — all as numpy + optional torch ops.
        Only division/mutation remains sequential."""
        env = self.env
        cfg = self.cfg
        n = len(alive)
        if n == 0:
            return []

        enable_3d = getattr(self, 'view_mode', '2d') == '3d'

        # ── 1. Extract agent arrays ──
        xs = np.array([a.x for a in alive], dtype=np.int32)
        ys = np.array([a.y for a in alive], dtype=np.int32)
        zs = np.array([a.z for a in alive], dtype=np.float32)
        ages = np.array([a.age for a in alive], dtype=np.int32)
        biomasses = np.array([a.biomass for a in alive], dtype=np.float64)
        phases = np.array([_PHASE_INT[a.phase] for a in alive], dtype=np.int32)
        gids = np.array([a.genotype.id for a in alive], dtype=np.int32)
        efficiencies = np.array([a.genotype.nutrient_efficiency for a in alive], dtype=np.float64)
        resistances = np.array([a.genotype.antibiotic_resistance for a in alive], dtype=np.float64)
        tox_prods = np.array([a.genotype.toxin_production for a in alive], dtype=np.float64)
        pg_prods = np.array([a.genotype.public_good_production for a in alive], dtype=np.float64)
        biofilm_members = np.array([a.biofilm_member for a in alive], dtype=bool)
        rl_acts = np.array([a._rl_action for a in alive], dtype=np.int32)
        rl_cooperate = np.array([a._rl_cooperate for a in alive], dtype=bool)
        rl_compete = np.array([a._rl_compete for a in alive], dtype=bool)

        # Pre-cache config constants
        mu_max_base = cfg["monod"]["mu_max"]
        Ks = cfg["monod"]["Ks"]
        yld = cfg["monod"]["yield_coefficient"]
        maint = cfg["bacterium"]["maintenance_energy"]
        div_thresh = cfg["bacterium"]["division_threshold"]
        max_age = cfg["bacterium"]["max_age"]
        lag_dur = cfg["bacterium"]["lag_phase_duration"]
        base_death = cfg["bacterium"]["base_death_rate"]
        tox_lethality = cfg["toxin"]["lethality"]
        tox_secr_rate = cfg["toxin"]["secretion_rate"]
        tox_cost_rate = cfg["bacterium"]["toxin_production_cost"]
        pg_cost_rate = cfg["bacterium"]["public_good_cost"]
        qs_prod = cfg["quorum_sensing"]["signal_production_rate"]
        qs_thresh = cfg["quorum_sensing"]["activation_threshold"]
        qs_bio_resist = cfg["quorum_sensing"]["biofilm_resistance_multiplier"]
        share_frac = cfg["quorum_sensing"].get("biofilm_resource_sharing", 0.0)
        carrying = cfg["population"]["carrying_capacity"]
        f_wg = cfg["fitness"]["weight_growth"]
        f_wr = cfg["fitness"]["weight_resistance"]
        f_we = cfg["fitness"]["weight_efficiency"]
        f_wc = cfg["fitness"]["weight_cooperation"]
        z_levels = max(cfg.get("grid", {}).get("z_levels", 10), 1)
        w_max = env.width - 1
        h_max = env.height - 1
        growth_mod = env.growth_modifier
        carrying_ratio = pop_size / carrying if carrying > 0 else 1.0

        # ── 2. Age increment ──
        ages += 1

        # ── 3. Phase transitions (vectorised) ──
        # LAG → LOG
        lag_mask = (phases == 0) & (ages >= lag_dur)
        phases[lag_mask] = 1
        # LOG → STATIONARY
        local_res = env.resource[ys, xs]
        log_to_stat = (phases == 1) & ((carrying_ratio >= 1.0) | (local_res < 0.05))
        phases[log_to_stat] = 2
        # STATIONARY → DEATH or back to LOG
        stat_mask = phases == 2
        stat_to_death = stat_mask & (local_res < 0.01)
        phases[stat_to_death] = 3
        stat_to_log = stat_mask & ~stat_to_death & (carrying_ratio < 0.8) & (local_res > 0.5)
        phases[stat_to_log] = 1

        # ── 4. Fitness (vectorised) ──
        mu_max_arr = mu_max_base * efficiencies
        ks_res = Ks + local_res
        growth_raw = np.where(ks_res > 0, mu_max_arr * local_res / ks_res, 0.0)
        norm_growth = np.minimum(1.0, growth_raw / max(mu_max_base, 0.01))
        local_ab = env.antibiotic[ys, xs]
        ab_penalty = local_ab * (1.0 - resistances) * 0.1
        effic_norm = np.minimum(1.5, efficiencies) / 1.5
        fitness = np.maximum(0.0,
            f_wg * norm_growth + f_wr * resistances + f_we * effic_norm + f_wc * pg_prods - ab_penalty
        )

        # ── 5. Maintenance metabolism ──
        biomasses -= maint
        np.maximum(biomasses, 0.0, out=biomasses)

        # ── 6. Movement (vectorised chemotaxis) ──
        can_move = (phases == 1) | (phases == 2)  # LOG or STATIONARY
        if can_move.any():
            movers = np.where(can_move)[0]
            n_movers = len(movers)
            mx, my = xs[movers], ys[movers]

            # Determine run bias per agent
            run_bias = np.full(n_movers, 0.6, dtype=np.float32)
            if rl_enabled:
                _RUN = int(BacterialAction.CHEMOTAXIS_RUN)
                _TUMBLE = int(BacterialAction.CHEMOTAXIS_TUMBLE)
                rl_m = rl_acts[movers]
                run_bias[rl_m == _RUN] = 0.9
                run_bias[rl_m == _TUMBLE] = 0.1

            is_run = np.random.random(n_movers) < run_bias
            # Tumble: random direction
            dirs_8 = np.array([(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1),(0,0)], dtype=np.int32)
            tumble_idx = np.random.randint(0, 9, size=n_movers)
            tumble_dx = dirs_8[tumble_idx, 0]
            tumble_dy = dirs_8[tumble_idx, 1]

            # Run: find best neighbour by resource gradient
            best_dx = np.zeros(n_movers, dtype=np.int32)
            best_dy = np.zeros(n_movers, dtype=np.int32)
            best_res = env.resource[my, mx].copy()
            for ddx, ddy in dirs_8:
                nx = np.clip(mx + ddx, 0, w_max)
                ny = np.clip(my + ddy, 0, h_max)
                r_nb = env.resource[ny, nx]
                better = r_nb > best_res
                best_dx[better] = ddx
                best_dy[better] = ddy
                best_res[better] = r_nb[better]

            # Merge run/tumble
            final_dx = np.where(is_run, best_dx, tumble_dx)
            final_dy = np.where(is_run, best_dy, tumble_dy)
            xs[movers] = np.clip(mx + final_dx, 0, w_max)
            ys[movers] = np.clip(my + final_dy, 0, h_max)

            # 3D z-axis movement
            if enable_3d:
                dz = np.random.choice([-1, 0, 0, 0, 1], size=n_movers).astype(np.float32)
                deep = zs[movers] > z_levels * 0.5
                surface_bias = np.random.random(n_movers) < 0.3
                dz[deep & surface_bias] = -1
                zs[movers] = np.clip(zs[movers] + dz, 0.0, float(z_levels))

        # ── 7. Growth (vectorised Monod) ──
        growing = (phases == 1) | (phases == 2)
        if growing.any():
            grow_idx = np.where(growing)[0]
            # Re-read resource at (possibly moved) positions
            local_res_g = env.resource[ys[grow_idx], xs[grow_idx]]
            ks_res_g = Ks + local_res_g
            mu = np.where(ks_res_g > 0, mu_max_base * efficiencies[grow_idx] * local_res_g / ks_res_g, 0.0)
            # Stationary-phase penalty
            is_stat = phases[grow_idx] == 2
            mu[is_stat] *= 0.1
            mu *= growth_mod
            # 3D depth modifier
            z_depth = zs[grow_idx] / z_levels
            mu *= (1.0 - 0.3 * z_depth)
            # RL CONSERVE
            conserving = rl_acts[grow_idx] == 5
            mu[conserving] *= 0.5
            # Resource consumption (mass-balance)
            substrate_needed = np.where(yld > 0, mu / yld, mu)
            available = env.resource[ys[grow_idx], xs[grow_idx]]
            consumed = np.minimum(available, np.maximum(0.0, substrate_needed))
            env.resource[ys[grow_idx], xs[grow_idx]] -= consumed
            env.total_resource_consumed += float(consumed.sum())
            biomasses[grow_idx] += consumed * yld

        # ── 8. Signal production (scatter-add) ──
        env.signal[ys, xs] += qs_prod

        # ── 9. Biofilm membership ──
        local_signal = env.signal[ys, xs]
        biofilm_members = local_signal >= qs_thresh

        # ── 10. Cooperation: EPS biofilm production ──
        coop_mask = biofilm_members & ((phases == 1) | (phases == 2))
        if coop_mask.any():
            ci = np.where(coop_mask)[0]
            eps_mult = np.where(rl_cooperate[ci], 2.0, 1.0)
            eps_amounts = pg_prods[ci] * 0.1 * eps_mult
            costs = eps_amounts * pg_cost_rate
            can_afford = biomasses[ci] > costs
            ci_ok = ci[can_afford]
            if len(ci_ok) > 0:
                biomasses[ci_ok] -= costs[can_afford]
                # scatter-add biofilm
                np.add.at(env.biofilm, (ys[ci_ok], xs[ci_ok]), (eps_amounts[can_afford]))

                # Resource sharing within biofilm
                if share_frac > 0:
                    local_r = env.resource[ys[ci_ok], xs[ci_ok]]
                    shareable = local_r > 0.1
                    if shareable.any():
                        s_idx = ci_ok[shareable]
                        shared = np.minimum(env.resource[ys[s_idx], xs[s_idx]], share_frac * local_r[shareable])
                        env.resource[ys[s_idx], xs[s_idx]] -= shared
                        share_each = shared / 8.0
                        for ddx in (-1, 0, 1):
                            for ddy in (-1, 0, 1):
                                if ddx == 0 and ddy == 0:
                                    continue
                                nx = np.clip(xs[s_idx] + ddx, 0, w_max)
                                ny = np.clip(ys[s_idx] + ddy, 0, h_max)
                                np.add.at(env.resource, (ny, nx), share_each)

        # ── 11. Competition: toxin secretion ──
        tox_mask = (phases == 1) | (phases == 2)
        if tox_mask.any():
            ti = np.where(tox_mask)[0]
            tox_mult = np.where(rl_compete[ti], 2.0, 1.0)
            tox_amounts = tox_secr_rate * tox_prods[ti] * tox_mult
            tox_costs = tox_amounts * tox_cost_rate
            can_secrete = biomasses[ti] > tox_costs
            ti_ok = ti[can_secrete]
            if len(ti_ok) > 0:
                biomasses[ti_ok] -= tox_costs[can_secrete]
                # scatter-add to per-genotype toxin grids
                for gid in np.unique(gids[ti_ok]):
                    g_mask = gids[ti_ok] == gid
                    g_idx = ti_ok[g_mask]
                    env.ensure_genotype_toxin_grid(int(gid))
                    np.add.at(env.toxin_grids[int(gid)], (ys[g_idx], xs[g_idx]),
                              tox_amounts[can_secrete][g_mask])

        # ── 12. Death (vectorised probability) ──
        age_factor = np.where(max_age > 0, (ages / max_age) ** 2, 0.0)
        local_res_d = env.resource[ys, xs]
        starvation = np.where(local_res_d < 0.01, 0.12,
                              np.where(local_res_d < 0.1, 0.03, 0.0))
        ab_d = env.antibiotic[ys, xs].copy()
        # Biofilm shield
        if biofilm_members.any():
            bio_idx = np.where(biofilm_members)[0]
            ab_d[bio_idx] *= qs_bio_resist
            biofilm_shield = env.biofilm[ys[bio_idx], xs[bio_idx]]
            ab_d[bio_idx] *= np.maximum(0.3, 1.0 - biofilm_shield * 0.1)
        effective_ab = ab_d * (1.0 - resistances)
        ab_death = 0.2 * effective_ab / (effective_ab + 2.0)

        # Foreign toxin
        foreign_tox = np.zeros(n, dtype=np.float64)
        for gid in np.unique(gids):
            g_mask = gids == gid
            g_idx = np.where(g_mask)[0]
            ft = np.zeros(len(g_idx), dtype=np.float64)
            for g, grid in env.toxin_grids.items():
                if g != int(gid):
                    ft += grid[ys[g_idx], xs[g_idx]]
            foreign_tox[g_idx] = ft
        tox_death = foreign_tox * tox_lethality

        phase_boost = np.where(phases == 3, 0.15, 0.0)
        density_press = np.maximum(0.0, (carrying_ratio - 0.8)) * 0.3
        biomass_death = np.where(biomasses <= 0, 0.2, 0.0)

        total_death_prob = np.minimum(0.95,
            base_death + age_factor + starvation + ab_death + tox_death
            + phase_boost + density_press + biomass_death
        )
        dies = np.random.random(n) < total_death_prob

        # ── 13. Write back to agents + handle division (sequential) ──
        new_agents: list[Bacterium] = []
        for i, agent in enumerate(alive):
            agent.x = int(xs[i])
            agent.y = int(ys[i])
            agent.z = float(zs[i])
            agent.age = int(ages[i])
            agent.biomass = float(biomasses[i])
            agent.phase = _INT_PHASE[int(phases[i])]
            agent.fitness = float(fitness[i])
            agent.biofilm_member = bool(biofilm_members[i])
            agent._carrying_ratio = carrying_ratio
            # Reset RL flags
            agent._rl_action = 6
            agent._rl_cooperate = False
            agent._rl_compete = False

            if dies[i]:
                agent.alive = False
                self.deaths_this_epoch += 1
                continue

            # Division (must remain sequential — creates new objects)
            if agent.biomass >= div_thresh and carrying_ratio < 1.0 and agent.phase in (Phase.LOG, Phase.STATIONARY):
                daughter = agent._divide(env)
                if daughter is not None:
                    new_agents.append(daughter)
                    self.divisions_this_epoch += 1
                    if daughter.mutated_this_division:
                        self.mutations_this_epoch += 1
            elif agent.biomass >= div_thresh and carrying_ratio >= 1.0:
                agent.biomass = div_thresh * 0.9

        return new_agents

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
