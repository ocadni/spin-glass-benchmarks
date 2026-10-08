**Success probability and TTS**

There are no proven ground states for the SK instances, so every row
(Greedy, Random, Reluctant, Population Annealing, Simulated Annealing and
Global Annealing) measures success against the same reference: the lowest
energy per spin found for that instance by any of these algorithms. A run
succeeds if its energy is within $1.5 \times 10^{-6}$ per spin of the
reference. Rows whose best energy is above the reference have success
probability 0 and an infinite TTS.

**Population Annealing, Simulated Annealing and Global Annealing**

All three run on one NVIDIA Tesla V100-SXM2-32GB, with a population of
$10^5$ replicas, a `logT` schedule from $T = 1.5$ to $T = 0.1$, 50
thermalization sweeps at $T = 1.5$, and a final $T = 0$ quench of the
population (single-spin flips accepted only if they lower the energy, until
none does). Each instance is run 10 times. A run's energy is the lowest one
it reaches, during the anneal or in the quench, and its runtime includes the
quench (and, for Global Annealing, the training of the MADE network).

| | Temperatures ($N \le 800$ / $N \ge 1000$) | Per temperature |
|---|---|---|
| Population Annealing | 10 / 20 | 10 local sweeps, multinomial resampling |
| Simulated Annealing | 100 / 200 | 1 local sweep |
| Global Annealing | 10 / 20 | 5 global (MADE) steps, each followed by 15 local sweeps |

Global Annealing trains the MADE network for 40 epochs at the start and 1
epoch at each new temperature (batch size 256, Adam, learning rate
$10^{-3}$).

The Parameters column abbreviates the `summary.csv` fields: $\theta_g$ is
`global_steps_per_temperature`, $\theta_l$ is `MCS_per_global_steps`
(Global Annealing) or `MCS_per_temperature` (Population and Simulated
Annealing), and Num. temp. is `number_of_temperatures`.
