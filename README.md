# QIGA: Quantum-Inspired vs Classical Genetic Algorithms

A small framework that compares a **quantum-inspired genetic algorithm (QIGA)** with a
**classical genetic algorithm** on two combinatorial problems, the **0/1 knapsack** and the
**travelling salesman problem (TSP)**, at an equal budget of fitness evaluations.

In QIGA each gene is a **Q-bit**: a pair of amplitudes `[α, β]` with `α² + β² = 1`.
Observing (measuring) a Q-bit gives `0` with probability `α²` and `1` with probability `β²`,
so a chromosome of Q-bits describes a probability distribution over solutions.
Every generation, QIGA observes concrete solutions and then rotates the Q-bits towards the
best solution seen so far.

The project contains:

* **`qiga`**: problems, algorithms, experiment runner, analysis and statistics;
* **`qiga_viz`**: figures for experiment results and traced runs, packaged separately so
  the core does not depend on matplotlib.

---

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Command line](#command-line)
4. [Experiment files (YAML)](#experiment-files-yaml)
5. [Algorithms](#algorithms)
6. [Problems](#problems)
7. [Metrics and statistics](#metrics-and-statistics)
8. [Figures](#figures)
9. [Python API](#python-api)
10. [Results](#results)
11. [Project layout](#project-layout)
12. [Development](#development)
13. [References](#references)

---

## Installation

Requires **Python ≥ 3.14**.

```bash
make init_venv       # creates ./dev_venv and installs the package (pip + uv)
make test_init       # + pytest and matplotlib (needed for the test suite)
```

Or by hand:

```bash
python3.14 -m venv dev_venv
dev_venv/bin/pip install -e .              # core: qiga
dev_venv/bin/pip install -e ".[viz]"       # + figures: qiga_viz
dev_venv/bin/pip install -e ".[tests]"     # + test suite
```

### Dependencies

| Group | File | Packages | Used for |
|---|---|---|---|
| core | `requirements/main.in` | `deap`, `numpy`, `pandas`, `pyarrow`, `pydantic`, `pyyaml`, `scipy` | classical GA operators (DEAP), Q-bit arithmetic, validated configs, results in Parquet, YAML experiments, significance tests |
| viz | `requirements/viz.in` | `matplotlib` (and Pillow, which it pulls in) | figures, GIF animation |
| tests | `requirements/tests.in` | viz + `pytest`, `pytest-cov` | test suite |
| linters | `requirements/linters.in` | `black`, `ruff`, `flake8`, … | code style |

The `.in` files list direct dependencies; the pinned `.txt` files are generated with
`uv pip compile requirements/<group>.in -o requirements/<group>.txt -n`
(or `make update_deps` to upgrade everything).

---

## Quick start

```bash
# 1. run a small experiment (knapsack n=100 + TSP n=30, 3 algorithms x 10 seeds, ~1 min)
make experiment SPEC=experiments/smoke.yaml WORKERS=4

# 2. summary table + all figures + statistics for the latest run
make report RESULTS=results/smoke

# 3. watch a single QIGA run: Q-bit heatmap, best tour, tour animation
make demo
```

Figures are written to `results/smoke/<timestamp>/figures/`; the demo writes to `results/demo/`.

The same without `make` (after `pip install -e ".[viz]"`):

```bash
python -m qiga run experiments/smoke.yaml -w 4     # prints results/smoke/<timestamp>
python -m qiga summary results/smoke               # summary table of the latest run
python -m qiga stats results/smoke -m anytime      # significance tests
python -m qiga_viz report results/smoke            # figures + CSV tables
python -m qiga_viz demo tsp -n 30 -e 10000         # traced QIGA run
```

The full study (about 2,700 runs, roughly 15–20 minutes on 8 cores):

```bash
make experiments_all WORKERS=8   # comparison, repair, sweep_knapsack, sweep_tsp
make reports_all                 # figures and statistics for all four
```

---

## Command line

### `python -m qiga`: run and analyse

| Command | Description |
|---|---|
| `python -m qiga run <spec.yaml> [-o results] [-w N]` | Run every algorithm on every problem for every seed, in `N` parallel processes. Prints the results folder. |
| `python -m qiga summary <results>` | Table per problem × algorithm: reference value, mean, std and median of the best fitness, mean final gap, mean anytime gap, hit rate and time. |
| `python -m qiga stats <results> [-m gap\|anytime\|target] [-t 1.0]` | Pairwise Mann-Whitney U tests (Holm-corrected) with the A12 effect size, plus mean ranks and the Friedman test across problems. |

`<results>` can be a single run folder or an experiment folder such as `results/smoke`,
in which case the latest run is used.

### `python -m qiga_viz`: figures

| Command | Description |
|---|---|
| `python -m qiga_viz report <results> [-o dir]` | All figures (PNG + PDF), `summary.csv` and the statistics CSVs. |
| `python -m qiga_viz demo {tsp,knapsack} [-n 30] [-a qiga] [-e 5000] [-s 0] [-o results/demo]` | Run one algorithm with tracing; writes the Q-bit heatmap (QIGA) and, for TSP, the best tour and a GIF of the tour improving. |

### Makefile shortcuts

| Target | Does |
|---|---|
| `make experiment SPEC=… WORKERS=4` | run one experiment |
| `make experiments_all` | run `comparison`, `repair`, `sweep_knapsack` and `sweep_tsp` |
| `make report RESULTS=…` | summary + figures for one results folder |
| `make reports_all` | reports for the four experiments above |
| `make stats RESULTS=… METRIC=anytime` | significance tests |
| `make demo` | traced QIGA run on TSP (n=30, 10k evaluations) |
| `make test` / `make check` / `make format` | tests / linters / formatting |
| `make help` | list all targets |

---

## Experiment files (YAML)

An experiment runs **every algorithm variant on every problem once per seed**. Every run
gets the same evaluation budget, so the comparison is fair whatever the population sizes.

```yaml
name: comparison              # results go to results/<name>/<UTC timestamp>/
seeds: 30                     # a count (0..29) or an explicit list [1, 7, 42]
max_evaluations: 30000        # equal budget for every run (generation limit is lifted)

problems:
  - {type: knapsack, n: 250, correlation: strongly_correlated, seed: 0}
  - {type: tsp, n: 50, seed: 0}

algorithms:
  - algorithm: qiga                          # Han & Kim defaults
  - algorithm: qiga
    label: qiga_tuned                        # a label makes variants of one algorithm distinct
    params: {epsilon: 0.01, delta_theta: 0.005pi}
  - algorithm: classical_ga
    params: {population_size: 50}
  - algorithm: random_search
    params: {population_size: 50}
```

* **`params`** are validated against the algorithm's config class (see [Algorithms](#algorithms)),
  so typos and out-of-range values fail before anything runs.
* Angles can be written as multiples of π: `0.01pi`, `0.01*pi`, `1e-3 pi`.
* **`sweep`** runs one variant per combination of values (the cartesian product):

  ```yaml
  - algorithm: qiga
    sweep:
      delta_theta: [0.005pi, 0.01pi, 0.05pi]
      bits_per_key: [4, 6, 8]          # 3 x 3 = 9 variants: qiga[delta_theta=0.005pi,bits_per_key=4], …
  ```

### Problem entries

| `type` | Fields (defaults) |
|---|---|
| `knapsack` | `n`, `correlation` (`uncorrelated`, `weakly_correlated` or `strongly_correlated`; default `uncorrelated`), `max_weight` (1000), `capacity_ratio` (0.5), `seed` (0) |
| `tsp` | `n`, `size` (1000: side of the square the cities are placed in), `rounded` (false: TSPLIB EUC_2D rounding), `seed` (0) |

### Included experiments

| File | What it shows | Runs |
|---|---|---|
| `experiments/smoke.yaml` | Quick end-to-end check | 60 |
| `experiments/comparison.yaml` | **Main result**: QIGA vs GA vs random search on 5 knapsack and 3 TSP instances, 30 seeds, 30k evaluations | 720 |
| `experiments/repair.yaml` | Knapsack with and without greedy repair | 360 |
| `experiments/sweep_knapsack.yaml` | QIGA sensitivity to Δθ, population size, H_ε and migration | 800 |
| `experiments/sweep_tsp.yaml` | QIGA on TSP: Δθ × bits per key, and H_ε | 800 |

### Results folder

```
results/<name>/<timestamp>/
├── experiment.yaml       # the validated spec that produced the results
├── runs.parquet          # one row per run: best fitness, evaluations, stop reason, time …
├── convergence.parquet   # every improvement of the best-so-far fitness (evaluations, fitness)
├── history.parquet       # per-generation population statistics (best, mean, std, worst)
└── figures/              # written by `qiga_viz report`
```

Runs are independent and seeded, so the results do not depend on the number of workers.

---

## Algorithms

All algorithms share one template (`EvolutionaryAlgorithm`): initialize → loop
`step` until the stop condition. Every fitness evaluation goes through a counting
`Evaluator`, which enforces the budget exactly (a generation cut short by the budget
still counts the solutions it evaluated) and records every improvement.

### Common parameters (`GAConfig`)

| Parameter | Default | Meaning |
|---|---|---|
| `population_size` | 50 | individuals per generation (QIGA: 10) |
| `generations` | 100 | generation limit (lifted when an experiment sets `max_evaluations`) |
| `max_evaluations` | – | evaluation budget |
| `target_fitness` | – | stop when reached |
| `seed` | – | RNG seed (set per run by the experiment) |
| `repair` | false | apply the problem's repair before evaluation; the repaired solution replaces the original (Lamarckian) |

### Classical GA (`classical_ga`, built on DEAP)

Generational GA with elitism. Operators are chosen by the problem's encoding unless set.

| Parameter | Default | Options |
|---|---|---|
| `elitism` | 1 | best individuals copied unchanged |
| `selection` | `tournament` | `tournament`, `roulette` |
| `tournament_size` | 3 | |
| `crossover` | binary `two_point`, permutation `ordered` | `one_point`, `two_point`, `uniform`, `ordered` (OX), `pmx` |
| `crossover_prob` | 0.9 | |
| `mutation` | binary `flip_bit`, permutation `inversion` | `flip_bit`, `shuffle`, `inversion` (a 2-opt move) |
| `mutation_prob` | 1.0 | chance an offspring is mutated |
| `gene_mutation_prob` | 1/n | per-gene rate for `flip_bit` / `shuffle` |

### QIGA (`qiga`, Han & Kim 2002)

Each individual *j* has a Q-chromosome *q_j* and an **attractor** *b_j*, the best solution
it has observed. All Q-bits start at `α = β = 1/√2`, so every solution is equally likely.
One generation:

1. **Observe** each *q_j* → bit string *x_j*; decode it into a solution, optionally repair it, and evaluate it.
2. **Rotate** every Q-bit where *x_j* and *b_j* differ, by `delta_theta` towards *b_j* when *x_j* is worse
   (the lookup table of Han & Kim; `delta_theta_better` towards *x_j* when it is at least as good, 0 in the paper).
3. Optional **gates**: quantum **NOT** (swaps α and β with probability `not_gate_prob`, a mutation) and
   **H_ε**, which keeps every probability inside `[ε, 1 − ε]` so no Q-bit freezes completely.
4. **Update** *b_j* ← *x_j* if *x_j* is at least as good.
5. **Migration**: every generation each *b_j* is replaced by the best attractor in its local
   group; every `global_migration_period` generations, by the global best.

There is no crossover or selection: information spreads only through the attractors.

| Parameter | Default | Meaning |
|---|---|---|
| `population_size` | 10 | Q-chromosomes |
| `delta_theta` | 0.01π | rotation angle towards the attractor (θ3 = θ5 in the paper) |
| `delta_theta_better` | 0 | rotation angle towards a better observation (θ4 = θ6) |
| `rotation_schedule` | `constant` | `linear_decay` shrinks the angles down to `delta_theta_min` |
| `delta_theta_min` | 0.001π | final angle for `linear_decay` |
| `local_group_size` | 2 | local migration group size (1 disables it) |
| `global_migration_period` | 100 | generations between global migrations (`null` disables it) |
| `epsilon` | 0 | H_ε gate: probabilities stay in `[ε, 1 − ε]` |
| `not_gate_prob` | 0 | per-Q-bit probability of a quantum NOT |
| `bits_per_key` | ⌈log₂ n⌉ + 1 | TSP only: Q-bits per city key |

#### From Q-bits to solutions: decoders

QIGA always works on bit strings; a **decoder** turns an observed bit string into a solution:

* **Knapsack**: `BinaryDecoder`, one Q-bit per item; the bits *are* the solution.
* **TSP**: `RandomKeyDecoder`. Each city gets `bits_per_key` Q-bits, read as an integer
  key (most significant bit first), and the tour visits the cities in order of their keys
  (Bean's random keys). Every bit string decodes to a valid tour; equal keys keep the cities'
  index order, so decoding is deterministic.

### Random search (`random_search`)

Uniformly random solutions in batches of `population_size`: the baseline any algorithm must beat.

---

## Problems

| Problem | Encoding | Objective | Infeasible solutions | Reference value |
|---|---|---|---|---|
| **0/1 knapsack** | binary, length n | maximize total value | over capacity → fitness 0; optional greedy repair (drops the worst value/weight-ratio items until the solution fits, then adds the best-ratio items that still fit) | exact optimum (dynamic programming) |
| **TSP** (closed tour) | permutation of n cities | minimize tour length | cannot occur | best tour found by any run in the experiment |

Knapsack instances follow Pisinger's classes: values uncorrelated with weights, weakly
correlated (`w ± R/10`) or strongly correlated (`w + R/10`). The more correlated, the harder.
Euclidean TSP instances place `n` cities uniformly at random in a square.

---

## Metrics and statistics

Everything is reported as a **gap** to the reference value, in percent (0 = reference reached),
so lower is better for both problems.

| Metric | Meaning | Answers |
|---|---|---|
| **final gap** | gap of the best solution at the end of the budget | which algorithm ends with better solutions |
| **anytime gap** | area under the best-so-far gap curve ÷ budget, i.e. the mean gap over the whole run | which algorithm finds good solutions **sooner** (convergence speed in one number) |
| **evaluations to target** | first evaluation at which the gap was ≤ t % | how fast a quality level is reached |

`python -m qiga stats` compares every pair of algorithms on each problem with a two-sided
**Mann-Whitney U** test (Holm-Bonferroni corrected per problem) and the **Vargha-Delaney A12**
effect size: the probability that a run of A is better than a run of B, where 0.5 is a tie and
|A12 − 0.5| ≥ 0.21 is a large effect. Across problems it reports each algorithm's mean rank and
the **Friedman** test.

---

## Figures

`python -m qiga_viz report` writes:

| File | Figure |
|---|---|
| `convergence_gap` / `convergence_fitness` | median best-so-far over the seeds (line) with the interquartile range (band), one panel per problem |
| `ecdf_targets` | runtime ECDF: fraction of (run, target) pairs reached within x evaluations, targets 10/5/2/1/0.5 %, per problem and over all problems |
| `distribution_gap` / `distribution_anytime` | box plot plus every run as a dot |
| `sweep_<parameter>` | final gap (median and IQR) against each swept parameter's values |
| `summary.csv`, `stats_pairwise_*.csv`, `stats_ranks_*.csv` | tables |

Colors follow the order of the algorithms in the YAML file, so keep that order the same
across experiments to keep the colors consistent. Line plots are skipped when an experiment
has more than 8 variants (a large sweep); the sweep and distribution plots cover those.

`python -m qiga_viz demo` adds:

| File | Figure |
|---|---|
| `qbits` | heatmap of the population-mean P(observe 1) of every Q-bit over the run: gray = undecided (½), blue → 1, orange → 0. It shows the Q-bits collapsing as QIGA converges. |
| `tour` / `tour.gif` | best tour found / animation of every improvement |

---

## Python API

```python
from qiga.algorithms import QIGA, ClassicalGA, run_traced
from qiga.config import QIGAConfig, ClassicalGAConfig
from qiga.problems import KnapsackKind, generate_euclidean_tsp, generate_knapsack

knapsack = generate_knapsack(250, KnapsackKind.STRONGLY_CORRELATED, seed=0)
result = QIGA(knapsack, QIGAConfig(max_evaluations=30_000, generations=10**9, epsilon=0.01, seed=1)).run()
print(result.best_fitness, knapsack.exact_optimum(), result.evaluations_to_target)

tsp = generate_euclidean_tsp(50, seed=0)
ga = ClassicalGA(tsp, ClassicalGAConfig(population_size=50, max_evaluations=30_000, generations=10**9, seed=1))
print(ga.run().best_fitness)

# trace a run for figures: best solution, and for QIGA the Q-bit probabilities, every 10 generations
result, trace = run_traced(QIGA(tsp, QIGAConfig(generations=2000, seed=1)), every=10)
```

```python
# experiments and analysis from code
from qiga.experiments.schema import ExperimentSpec
from qiga.experiments.runner import run_experiment
from qiga.experiments.analysis import load_results, summary_table, run_metrics
from qiga.experiments.stats import pairwise_tests, friedman_ranks
from qiga_viz import plot_convergence, plot_target_ecdf

folder = run_experiment(ExperimentSpec.from_yaml("experiments/smoke.yaml"), "results", workers=4)
results = load_results(folder)
print(summary_table(results))
print(pairwise_tests(results, "anytime"))
plot_convergence(results).savefig("convergence.png", dpi=200)
```

### Extending

* **New problem**: subclass `qiga.problems.base.Problem` (a frozen pydantic model): set `sense` and
  `encoding`, implement `dimension`, `evaluate`, `is_feasible` and `random_solution`, and optionally
  `repair` and `known_optimum`. Add a `*Spec` class in `experiments/schema.py` to use it from YAML.
* **New algorithm**: subclass `EvolutionaryAlgorithm[YourConfig]`, implement `_initialize`, `_step`
  and `_population_fitness`, evaluate only through `self.evaluate` and draw randomness only from
  `self.rng`. Register it in `qiga/experiments/registry.py`.
* **New encoding for QIGA**: subclass `qiga.representations.decoders.Decoder` and add it to `make_decoder`.

---

## Results

From `comparison.yaml` and `repair.yaml` (30 seeds, 30,000 evaluations each, mean final gap).
Re-run with `make experiments_all && make reports_all`.

| Instance | QIGA (paper defaults) | Classical GA | Random search |
|---|---|---|---|
| knapsack, strongly correlated, n=100 | 2.04 % | **0.99 %** | 3.81 % |
| knapsack, strongly correlated, n=250 | 2.95 % | **1.40 %** | 4.60 % |
| knapsack, strongly correlated, n=500 | 3.65 % | **2.02 %** | 5.06 % |
| knapsack, uncorrelated, n=250 | 3.15 % | **0.74 %** | 27.57 % |
| knapsack, weakly correlated, n=250 | 3.74 % | **1.52 %** | 6.78 % |
| TSP n=30 (gap to the best tour found) | 42.2 % | **14.8 %** | 133.7 % |
| TSP n=50 | 34.6 % | **9.5 %** | 147.6 % |
| TSP n=100 | 20.3 % | **5.8 %** | 130.2 % |

* **With the paper's settings the GA wins on every instance**, on both final gap and anytime gap
  (Holm-corrected p < 0.001, large effects). QIGA always beats random search clearly.
* **QIGA converges prematurely.** Its curves keep up with the GA for the first ~5k evaluations
  and then flatten; the Q-bit heatmap shows the probabilities collapsing to 0 or 1, after which
  every observation is nearly the same solution.
* **Keeping the Q-bits open closes most of the gap.** On knapsack n=250 (median final gap,
  uncorrelated / strongly correlated): defaults 3.07 / 3.02 %, H_ε with ε = 0.005–0.01
  **1.0–1.1 / 1.6–1.9 %**, Δθ = 0.005π 1.8 / 2.3 %, global migration every 500 generations
  1.5 / 2.6 %; turning global migration off is much worse (9.8 / 3.9 %).
* **On TSP, H_ε helps as well, but the GA stays ahead.** Median tour length, n = 30 / 50:
  QIGA defaults 6640 / 11424, QIGA with ε = 0.005–0.01 (Δθ = 0.05π) **6022–6521 / 10228–10352**,
  classical GA **5532 / 9294**. The number of bits per key (4, 6 or 8) makes little difference,
  and Δθ has no clear trend. The GA's tour-aware operators (order crossover, inversion = 2-opt move)
  suit TSP better than rotating random-key bits, where one flipped bit can move a city anywhere in the tour.
* **With greedy repair** both algorithms get within 0.1 % of the optimum, with no significant
  difference in final quality; the GA still gets there sooner.
* TSP gaps are relative to the best tour any run found, not the true optimum, so compare them
  only between algorithms on the same instance.

---

## Project layout

```
src/qiga/
├── config.py                 # GAConfig, ClassicalGAConfig, QIGAConfig (pydantic)
├── problems/                 # Problem base, Knapsack, TSP, instance generators
├── representations/          # Q-bit gates, QBit, QChromosome, decoders (binary, random keys)
├── operators/                # DEAP operator registry, Han & Kim rotation table
├── algorithms/               # base template, Evaluator, ClassicalGA, QIGA, RandomSearch, tracing
├── experiments/              # YAML schema + sweeps, parallel runner, analysis, statistics
└── __main__.py               # python -m qiga run | summary | stats
src/qiga_viz/
├── style.py                  # palette, panel layout, labels
├── convergence.py  ecdf.py  distribution.py  sweep.py  trace.py
└── __main__.py               # python -m qiga_viz report | demo
experiments/                  # experiment YAML files
tests/                        # pytest suite (~350 tests, 99 % coverage)
requirements/                 # *.in (direct) and *.txt (pinned) dependency files
```

---

## Development

```bash
make test            # pytest with coverage (htmlcov/)
make check           # flake8, pylint, bandit, ruff on src/
make format          # black
```

Code style: black (line length 150), ruff with all rules enabled, flake8. All randomness goes
through seeded `numpy.random.Generator`s, so every run is reproducible.

---

## References

* K.-H. Han, J.-H. Kim. *Quantum-inspired evolutionary algorithm for a class of combinatorial
  optimization.* IEEE Transactions on Evolutionary Computation 6(6), 2002. (Q-bit representation,
  rotation lookup table, migration; knapsack setup.)
* K.-H. Han, J.-H. Kim. *Quantum-inspired evolutionary algorithms with a new termination criterion,
  H_ε gate, and two-phase scheme.* IEEE Transactions on Evolutionary Computation 8(2), 2004.
* J. C. Bean. *Genetics and random keys for sequencing and optimization.* ORSA Journal on Computing 6(2), 1994.
* D. Pisinger. *Where are the hard knapsack problems?* Computers & Operations Research 32(9), 2005.
* A. Vargha, H. D. Delaney. *A critique and improvement of the CL common language effect size
  statistics of McGraw and Wong.* Journal of Educational and Behavioral Statistics 25(2), 2000.
* F.-A. Fortin et al. *DEAP: Evolutionary algorithms made easy.* JMLR 13, 2012.
