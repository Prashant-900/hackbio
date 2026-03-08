"""
mesa_model.py — Mesa framework integration for the bacterial colony ABM.

Provides a genuine Mesa Model wrapping the Simulation engine, enabling
use with Mesa's DataCollector, batch runners, and parameter sweeps.

Usage:
    from mesa_model import BacterialColonyModel
    model = BacterialColonyModel(cfg)
    for _ in range(200):
        model.step()
    df = model.datacollector.get_model_vars_dataframe()

References:
  Mesa ABM framework: https://mesa.readthedocs.io/
  Kazil J et al. (2020) Utilizing Python for Agent-Based Modeling, JASSS.
"""

from __future__ import annotations

import mesa

from simulate import Simulation


class BacterialColonyModel(mesa.Model):
    """Mesa-compatible Model wrapping the bacterial colony simulation.

    Delegates all simulation logic to our ``Simulation`` class while
    providing Mesa's ``DataCollector`` interface for metrics gathering
    and batch-run analysis.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.sim = Simulation(cfg)
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Population": lambda m: m._latest("total_population", 0),
                "MeanFitness": lambda m: m._latest("mean_fitness", 0),
                "MeanResistance": lambda m: m._latest("mean_resistance", 0),
                "CooperationIndex": lambda m: m._latest("cooperation_index", 0),
                "CompetitionIndex": lambda m: m._latest("competition_index", 0),
                "BiofilmFraction": lambda m: m._latest("biofilm_fraction", 0),
                "ResourceConcentration": lambda m: m._latest(
                    "resource_concentration", 0
                ),
                "MutationFrequency": lambda m: m._latest("mutation_frequency", 0),
                "CumulativeMutations": lambda m: m._latest(
                    "cumulative_mutations", 0
                ),
                "CumulativeHGT": lambda m: m._latest("cumulative_hgt", 0),
                "MeanAntibiotic": lambda m: m._latest("mean_antibiotic", 0),
                "Epoch": lambda m: m.sim.epoch,
            }
        )
        self.datacollector.collect(self)

    # ── helpers ──────────────────────────────────────────────
    def _latest(self, key: str, default=0):
        if self.sim.metrics:
            return self.sim.metrics[-1].get(key, default)
        return default

    # ── Mesa interface ───────────────────────────────────────
    def step(self):
        self.sim.step()
        self.datacollector.collect(self)

    @property
    def epoch(self):
        return self.sim.epoch

    @property
    def env(self):
        return self.sim.env

    @property
    def metrics(self):
        return self.sim.metrics

    @property
    def bacteria(self):
        return [a for a in self.sim.agents if a.alive]

    def export_csv(self):
        return self.sim.export_csv()

    def run(self, epochs: int | None = None, callback=None):
        """Run the full simulation, collecting Mesa metrics each step."""
        total = epochs or self.sim.cfg["simulation"]["epochs"]
        for _ in range(total):
            self.step()
            if callback:
                callback(self.sim)
            if not any(a.alive for a in self.sim.agents):
                break
