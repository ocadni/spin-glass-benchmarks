**Success probability and TTS**

There are no proven ground states for the SK instances, so every row
(Greedy, Random, Reluctant, Population Annealing and Simulated Annealing)
measures success against the same reference: the lowest energy per spin
found for that instance by any of our runs. This includes runs not listed
here, such as scans over the number of temperatures with up to 100 runs per
instance and Global Annealing runs. A run succeeds if its energy is within
$1.5 \times 10^{-6}$ per spin of the reference. Rows whose best energy is
above the reference have success probability 0 and an infinite TTS.

**Population Annealing and Simulated Annealing**

Both run on one NVIDIA Tesla V100-SXM2-32GB, with a population of $10^5$
replicas (independent replicas for Simulated Annealing), a `logT` schedule
from $T = 1.5$ to $T = 0.1$, 50 thermalization sweeps at $T = 1.5$, and a
final $T = 0$ quench of the population (single-spin flips accepted only if
they lower the energy, until none does). Population Annealing uses
multinomial resampling. Each instance is run 10 times. A run's energy is the
lowest one it reaches, during the anneal or in the quench, and its runtime
includes the thermalization and the quench.

**Number of temperatures**

For each $N$, each algorithm was run at 6 to 13 numbers of temperatures on
all 100 instances. Each row reports, for that instance, the number of
temperatures with the lowest TTS (tuned per instance, as in the Parameters
column). An instance that an algorithm never solves reports the number of
temperatures that gave its lowest energy. Numbers of temperatures run:

| $N$ | Population Annealing (1 sweep per temperature) | Simulated Annealing (10 sweeps per temperature) |
|---|---|---|
| 200 | 6, 7, 9, 11, 13, 17, 25 | 100, 1 sweep per temperature$^*$ |
| 300 | 8, 11, 13, 15, 19, 25, 37 | 100, 1 sweep per temperature$^*$ |
| 400 | 13, 17, 21, 26, 31, 41, 61 | 2–8, 10 |
| 600 | 19, 26, 31, 36, 46, 61, 91 | 3–12, 14, 16, 20 |
| 800 | 26, 36, 46, 56, 71, 91, 136 | 4–10, 12, 14, 17 |
| 1000 | 31, 41, 51, 61, 76, 101, 151 | 3–12, 14, 16, 20 |
| 1200 | 41, 51, 66, 81, 101, 131, 196 | 6, 8, 10–14, 16, 19, 24 |
| 1400 | 51, 66, 81, 96, 121, 161, 241 | 7, 9, 11, 13–15, 17, 20, 24, 29 |
| 1600 | 61, 81, 101, 121, 151, 201, 301 | 10, 12, 14, 16, 20, 26$^\ddagger$ |
| 2000 | 71, 96, 121, 146, 181, 241, 361 | 9, 12, 15, 17, 19, 21, 23, 26, 30, 36$^\ddagger$ |

$^*$ For $N \le 100$ (both algorithms) and for Simulated Annealing at
$N = 200$ and $300$, the rows come from earlier runs with the same settings
but a single protocol for all instances: Population Annealing with 10
temperatures and 10 sweeps per temperature, Simulated Annealing with 100
temperatures and 1 sweep per temperature. Every one of these runs reaches
the reference.

$^\ddagger$ For most instances the best number of temperatures is the largest
one run, so the Simulated Annealing TTS at $N = 1600$ and $2000$ is mostly an
upper bound.

The sweeps per temperature were chosen from tests at $N = 600$, $1000$ and
$1400$. At the same total number of sweeps, Population Annealing with 1 sweep
per temperature succeeded more often than with 10 sweeps per temperature and
fewer temperatures. Simulated Annealing succeeded equally often with 1 or 10
sweeps per temperature, and 10 sweeps per temperature is faster per run.

The Parameters column abbreviates the `summary.csv` fields: $\theta_l$ is
`MCS_per_temperature` and Num. temp. is `number_of_temperatures`.
