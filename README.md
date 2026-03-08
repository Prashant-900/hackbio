# Bacterial Colony Dynamics — Agent-Based Model

A spatially-explicit agent-based simulation of bacterial colony growth, evolution, and survival under antibiotic stress. Validated against the **E. coli Long-Term Evolution Experiment (LTEE)** — the longest-running evolution experiment in biology (Lenski, 1988–present). Built for the IIT Mandi BioHack computational biology track.

**Live Demo →** [http://52.66.196.251/](http://52.66.196.251/)

---

## What It Does

Hundreds of individual bacteria live on a 2D grid with 3D colony depth. Each one grows, moves, divides, mutates, cooperates, competes, and dies — all governed by real microbiology equations. A **Deep Q-Network (DQN)** acts as a shared "bacterial brain," learning optimal survival strategies from pooled experience. An antibiotic front sweeps in from one edge; the colony must evolve resistance or perish.

The simulation models:

- **Growth** — Monod kinetics with yield-corrected substrate consumption
- **Life cycle** — Lag → Log → Stationary → Death phases (logistic S-curve)
- **Spatial diffusion** — nutrients, antibiotics, quorum signals (Fick's law, Neumann boundaries)
- **Mutation & evolution** — per-division trait perturbation with cumulative tracking (LTEE-style clock-like accumulation)
- **Cooperation** — quorum-sensing biofilm formation (LuxI/LuxR analogue)
- **Competition** — bacteriocin toxin warfare between genotypes
- **Horizontal gene transfer** — one-way conjugative plasmid transfer (donor → recipient, Frost et al. 2005)
- **Natural selection** — multi-trait fitness landscape
- **Chemotaxis** — run-and-tumble on full Moore neighbourhood (8 directions + stay)
- **Reinforcement Learning** — Double DQN with experience replay; 14-dim state, 7 discrete actions
- **3D Colony Structure** — z-coordinate depth per bacterium (biofilm layering)
- **Physics** — Cardinal temperature, pH, and pressure growth models (Rosso et al. 1993, 1995)
- **GPU Acceleration** — PyTorch CUDA/MPS for batch RL inference
- **Mesa Framework** — `BacterialColonyModel` wrapper with `DataCollector` for batch analysis

---

## Reinforcement Learning — Double DQN

A shared Deep Q-Network learns survival strategies that bacteria would evolve over millions of years, compressed into hundreds of training epochs.

### Architecture

- **Double DQN** with experience replay (Mnih et al. 2015; van Hasselt et al. 2016)
- **One shared brain** trained from all bacteria's pooled experience — computationally tractable even with 10,000+ agents
- **GPU-accelerated** batch inference via PyTorch (CUDA / MPS / CPU fallback)

### State Space (14 dimensions)

| Dim | Feature | Normalization |
|-----|---------|---------------|
| 0 | Local resource | S / S_max |
| 1 | Local antibiotic | AB / AB_max |
| 2 | Foreign toxin | toxin / 5.0 |
| 3 | QS signal | signal / 2.0 |
| 4 | Biofilm density | biofilm / 5.0 |
| 5 | Biomass | biomass / 3.0 |
| 6 | Age fraction | age / max_age |
| 7 | Growth phase | phase_int / 3 |
| 8 | Resistance | [0, 1] |
| 9 | Fitness | [0, 1] |
| 10 | Temperature | (T - 10) / 35 |
| 11 | Pressure | (P - 0.5) / 5 |
| 12 | z-depth | z / z_levels |
| 13 | Biofilm member | 0 or 1 |

### Action Space (7 discrete)

| Action | Effect |
|--------|--------|
| CHEMOTAXIS_RUN | Biased move toward nutrient gradient (run_bias=0.9) |
| CHEMOTAXIS_TUMBLE | Random direction move (run_bias=0.1) |
| COOPERATE | Double EPS biofilm secretion |
| COMPETE | Double bacteriocin production |
| CONJUGATE | Attempt HGT with neighbour |
| CONSERVE | Halve growth rate to save energy |
| GROW | Default metabolic behaviour |

### Reward Function

$$R = +0.1_{\text{alive}} + 2 \cdot \Delta m_{\text{biomass}} + 5_{\text{division}} + 0.3_{\text{biofilm}} - 10_{\text{death}}$$

### Network

```
Input(14) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(7)
```

Soft target-network updates (τ = 0.005), epsilon-greedy exploration (1.0 → 0.05), gradient clipping at 1.0.

---

## 3D Colony Structure

Bacteria have a z-coordinate representing depth in the colony:
- Daughters inherit parent's z ± noise, clamped to [0, z_levels]
- Deeper bacteria have **reduced nutrient access** (factor: 1 − 0.3 · z/z_levels)
- Surface bacteria face more antibiotic exposure
- The 3D structure is visualized as a Plotly scatter3d chart in the dashboard

---

## Physics — Cardinal Growth Models

Three environmental factors modulate growth rate multiplicatively:

$$\text{growth\_modifier} = f_T(T) \times f_{\text{pH}}(\text{pH}) \times f_P(P)$$

### Cardinal Temperature Model (Rosso et al. 1993)

$$f_T = \frac{(T - T_{\max})(T - T_{\min})^2}{(T_{\text{opt}} - T_{\min})[(T_{\text{opt}} - T_{\min})(T - T_{\text{opt}}) - (T_{\text{opt}} - T_{\max})(T_{\text{opt}} + T_{\min} - 2T)]}$$

| Parameter | Default | Unit |
|-----------|---------|------|
| T_min | 10 | °C |
| T_opt | 37 | °C |
| T_max | 45 | °C |

### Cardinal pH Model (Rosso et al. 1995)

$$f_{\text{pH}} = \frac{(\text{pH} - \text{pH}_{\min})(\text{pH} - \text{pH}_{\max})}{(\text{pH} - \text{pH}_{\min})(\text{pH} - \text{pH}_{\max}) - (\text{pH} - \text{pH}_{\text{opt}})^2}$$

Range: pH 4 – 9, optimal at 7.0.

### Pressure Model (Abe & Horikoshi 2001)

$$f_P = \max(0,\; 1 - (P - 1) / 500)$$

Growth declines linearly above 1 atm; full inhibition at ~500 atm.

All three factors can be adjusted in real time via the dashboard Settings panel.

---

## GPU Acceleration

- Auto-detects **NVIDIA CUDA**, **Apple MPS**, or falls back to **CPU**
- Dashboard shows GPU badge with device name and memory
- "Force CPU" toggle in settings to override GPU detection
- Batch inference: all alive bacteria's states are extracted as a single numpy array, converted to a torch tensor, and processed on GPU in one forward pass

---

## Mesa Framework Integration

`mesa_model.py` provides a `BacterialColonyModel(mesa.Model)` wrapper:

```python
from mesa_model import BacterialColonyModel
import yaml

cfg = yaml.safe_load(open('config.yaml'))
model = BacterialColonyModel(cfg)
for _ in range(200):
    model.step()

# Mesa DataCollector gives a clean pandas DataFrame
df = model.datacollector.get_model_vars_dataframe()
print(df[['Population', 'MeanFitness', 'CooperationIndex']].tail())
```

12 model reporters: Population, MeanFitness, MeanResistance, CooperationIndex, CompetitionIndex, BiofilmFraction, ResourceConcentration, MutationFrequency, CumulativeMutations, CumulativeHGT, MeanAntibiotic, Epoch.

---

## Core Equations

**Monod growth rate** — how fast a bacterium grows given local nutrients:

$$\mu = \mu_{\max} \cdot \frac{S}{K_s + S}$$

| Symbol | Meaning | Default |
|--------|---------|---------|
| $\mu_{\max}$ | Max growth rate | 0.8 h⁻¹ |
| $S$ | Local substrate concentration | — |
| $K_s$ | Half-saturation constant | 1.0 |

**Yield-corrected substrate consumption** (Monod 1949, Herbert 1958):

$$\Delta S = \frac{\mu}{Y_{X/S}}, \quad \Delta X = \Delta S \cdot Y_{X/S}$$

where $Y_{X/S} = 0.4$ is the yield coefficient (grams biomass per gram substrate). This ensures mass balance: 1 unit of growth consumes $1/Y = 2.5$ units of substrate.

**Logistic population growth** — the colony follows the classic S-curve (consistent with lab_bacterial_growth OD600 data):

$$N(t) = \frac{K}{1 + \frac{K - N_0}{N_0} \cdot e^{-kt}}$$

where $K$ = carrying capacity (10,000), $N_0$ = initial population (300), $k = \ln 2 / T_d$.

**Fitness** — weighted sum of traits driving natural selection:

$$F = 0.4 \cdot g + 0.3 \cdot r + 0.2 \cdot e + 0.1 \cdot c - \text{AB penalty}$$

where $g$ = normalized growth, $r$ = antibiotic resistance, $e$ = nutrient efficiency, $c$ = cooperation.

**Antibiotic death** — saturating Hill function (prevents instant wipeout):

$$P_{\text{ab}} = \frac{0.2 \cdot \text{AB}_{\text{eff}}}{\text{AB}_{\text{eff}} + 2.0}$$

**Diffusion** — discretized Fick's second law with no-flux (Neumann) boundaries:

$$C_{t+1} = C_t + D \cdot (\bar{C}_{\text{neighbors}} - C_t)$$

**Total death probability** per epoch:

$$P_{\text{death}} = P_{\text{base}} + P_{\text{age}} + P_{\text{starve}} + P_{\text{ab}} + P_{\text{toxin}} + P_{\text{density}} + P_{\text{phase}}$$

---

## Results

Default run: 200 epochs, 200×200 grid, 300 initial bacteria, seed 42.

### Population & Genotype Dynamics
![Evolution Curves](charts/evolution_curves.png)

### Genotype Frequency Over Time
![Genotype Frequency](charts/genotype_frequency.png)

### Resource & Antibiotic Dynamics
![Resource Dynamics](charts/resource_dynamics.png)

### Phase Distribution
![Phase Distribution](charts/phase_distribution.png)

### Cooperation vs Competition
![Cooperation Competition](charts/cooperation_competition.png)

### Fitness Evolution
![Fitness Evolution](charts/fitness_evolution.png)

### Spatial Colony Density (final epoch)
![Spatial Density](charts/spatial_colony_density.png)

### Spatial Antibiotic Gradient
![Spatial Antibiotic](charts/spatial_antibiotic.png)

### Spatial Genotype Map
![Spatial Genotype](charts/spatial_genotype_map.png)

---

## LTEE Validation

This simulation was validated against key findings from the **E. coli Long-Term Evolution Experiment** (Lenski et al., 1988–present), which has tracked 12 populations of *E. coli* for over 80,000 generations. We also cross-referenced the [`lab_bacterial_growth`](https://github.com/sgroverbiern/lab_bacterial_growth) reference implementation (logistic growth from OD600 spectrophotometer data).

### What the LTEE shows vs what our simulation reproduces

| LTEE Finding | Our Simulation | Status |
|---|---|---|
| Population follows logistic S-curve growth | 300 → 10,605 (logistic curve with clear lag, log, stationary phases) | ✅ |
| Mutations accumulate linearly (clock-like, ~1 per 300 generations) | 0 → 275 → 660 → 1,057 → 1,445 over 200 epochs (~7.2/epoch, linear $R^2 \approx 1$) | ✅ |
| Power-law fitness increase (rapid early gains, diminishing returns) | Mean fitness stabilizes ~0.50 after initial transient | ✅ |
| HGT is one-way conjugative transfer (donor retains plasmid) | Implemented: recipient acquires max(own, donor) resistance; donor unchanged | ✅ |
| Cooperation can evolve (cross-feeding, biofilm communities) | Biofilm fraction reaches 84%; cooperation index 0.40 | ✅ |
| Yield coefficient governs mass balance ($Y_{X/S} = \Delta X / \Delta S$) | Corrected formula: $\Delta S = \mu/Y$, $\Delta X = \Delta S \cdot Y$; total consumed = 2.3M units | ✅ |

### Biological fixes applied during validation

1. **HGT direction** — Changed from bidirectional trait swap to one-way conjugative transfer (Frost et al. 2005)
2. **Yield coefficient** — Corrected inverted formula that gave effective Y=2.5 instead of configured Y=0.4
3. **Chemotaxis** — Expanded from 4 cardinal directions to full Moore neighbourhood (8 + stay)
4. **QS biofilm** — Tuned activation threshold from 1.0 → 0.15 (old value was unreachable at typical cell density)
5. **Cumulative tracking** — Added LTEE-style running totals for mutations and HGT events
6. **Mass balance** — Added total resource consumed tracking for stoichiometric verification

---

## Configuration

All parameters live in [`config.yaml`](config.yaml). Key settings for the default run:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Grid | 200 × 200 | Spatial arena |
| Initial bacteria | 300 | Randomly placed |
| Carrying capacity | 10,000 | Logistic ceiling |
| Epochs | 200 | Simulation length |
| μ_max | 0.8 | Monod max growth rate |
| K_s | 1.0 | Half-saturation constant |
| Yield (Y_{X/S}) | 0.4 | Biomass per substrate consumed |
| Initial resource | 15.0 | Per-cell starting concentration |
| Replenishment rate | 0.25 | Substrate added per epoch |
| Max resource | 30.0 | Concentration cap |
| Mutation rate | 0.01 | Per-division probability |
| HGT probability | 0.005 | Conjugation frequency (one-way) |
| QS signal production | 0.05 | Per-bacterium per epoch |
| QS activation threshold | 0.15 | Biofilm formation trigger |
| Biofilm shield | 0.5× | AB penetration reduction |
| Antibiotic start | Epoch 60 | Gradual from top edge |
| Antibiotic decay | 0.015 | First-order degradation |
| AB max | 8.0 | Concentration cap |
| Seed | 42 | Reproducible |

---

## Project Structure

```
├── config.yaml        # All simulation parameters (physics, RL, grid, biology)
├── environment.py     # 2D grids + physics growth modifiers
├── agent.py           # Bacterium agent: growth, death, mutation, HGT, RL actions, z-depth
├── simulate.py        # Epoch loop, DQN integration, metrics collection, CSV export
├── rl_agent.py        # Double DQN: network, replay buffer, state/action/reward
├── gpu_utils.py       # CUDA/MPS/CPU device detection
├── mesa_model.py      # Mesa Model wrapper with DataCollector
├── visualize.py       # 15 matplotlib/seaborn charts
├── dashboard.py       # Flask + SocketIO live dashboard server
├── main.py            # CLI entry point
├── templates/
│   └── index.html     # Interactive dashboard UI (Canvas + Plotly + 3D)
├── Dockerfile         # Container deployment
├── requirements.txt   # Python dependencies (numpy, scipy, torch, mesa, flask...)
└── TEAM.txt           # Team info
```

---

## Local Setup

### Requirements

- Python 3.12+
- pip

### Install & Run

```bash
# Clone
git clone <repo-url>
cd hackbio

# Virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
.\venv\Scripts\Activate.ps1     # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Run headless simulation (generates CSV + 15 charts)
python main.py --epochs 200 --seed 42

# Run live dashboard
python dashboard.py
# Open http://localhost:5000
```

### CLI Options

```bash
python main.py --epochs 300 --seed 42 --initial-count 500 --carrying-capacity 15000
python main.py --dashboard    # launches the web UI instead
```

---

## Docker

```bash
# Build
docker build -t hackbio .

# Run
docker run -p 5000:5000 hackbio

# Open http://localhost:5000
```

---

## Live Dashboard Features

- **2D Canvas world** — zoom, pan, hover over individual bacteria
- **3D Colony chart** — Plotly scatter3d showing colony depth structure
- **Real-time stats** — population, fitness, resistance, cooperation, growth modifier
- **RL stats panel** — epsilon, loss, buffer size, device (GPU/CPU)
- **GPU indicator** — auto-detected GPU badge in topbar
- **Layer toggles** — resource, antibiotic, biofilm, signal overlays
- **11 live charts** — population, genotypes, phases, demographics, fitness, 3D colony
- **Physics controls** — temperature, pressure, pH sliders in settings
- **RL controls** — enable/disable RL brain, force CPU toggle
- **Settings panel** — adjust grid size, epochs, mutation rate, antibiotic mode
- **Speed control** — adjust simulation delay per epoch
- **Report download** — ZIP with 15 PNGs + CSV + config.yaml

---

## Team

**HackBio** — Prashant Suthar

Computational Biology / Agent-Based Modeling Track

---

## References

- Lenski RE et al. (1991). *Long-term experimental evolution in E. coli*. Am. Nat. — The foundational LTEE paper.
- Blount ZD, Borland CZ, Lenski RE (2008). *Historical contingency and the evolution of a key innovation in an experimental population of E. coli*. PNAS. — Citrate utilization, 3-step innovation model (Potentiation → Actualization → Refinement).
- Wiser MJ, Ribeck N, Lenski RE (2013). *Long-term dynamics of adaptation in asexual populations*. Science. — Power-law fitness trajectory in LTEE.
- Monod J. (1949). *The growth of bacterial cultures*. Ann. Rev. Microbiol.
- Herbert D, Elsworth R, Telling RC (1956). *The continuous culture of bacteria; a theoretical and experimental study*. J. Gen. Microbiol. — Yield coefficient $Y_{X/S}$.
- Riley MA, Wertz JE (2002). *Bacteriocins: evolution, ecology, and application*. Ann. Rev. Microbiol.
- Fuqua WC et al. (1994). *Quorum sensing in bacteria: the LuxR-LuxI family*. J. Bacteriol.
- Frost LS et al. (2005). *Mobile genetic elements: the agents of open source evolution*. Nat. Rev. Microbiol. — One-way conjugative HGT.
- Fisher RA (1930). *The Genetical Theory of Natural Selection*.
- Mnih V et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- van Hasselt H et al. (2016). *Deep Reinforcement Learning with Double Q-learning*. AAAI.
- Rosso L et al. (1993). *An unexpected correlation between cardinal temperatures of microbial growth*. J. Theor. Biol.
- Rosso L et al. (1995). *Convenient model to describe the combined effects of temperature and pH on microbial growth*. Appl. Environ. Microbiol.
- Abe F, Horikoshi K (2001). *The biotechnological potential of piezophiles*. Trends Biotechnol.
- Kazil J et al. (2020). *Utilizing Python for Agent-Based Modeling: The Mesa Framework*. JASSS.
