**Success probability and TTS**

For $N = 1000$ ($L = 10$) the reference energy is the verified ground state.
For $N = 2744$ ($L = 14$) it is the lowest energy found by earlier long Global
Annealing runs, which is not verified. The reference energies have 5
decimals, so a run succeeds if its energy per spin is within
$5.5 \times 10^{-6}$ of the reference. Rows that reach the reference report
the reference energy. No run went below the reference. Rows whose best energy
is above the reference (all Greedy, Random and Reluctant rows) have success
probability 0 and an infinite TTS.

**Population Annealing, Simulated Annealing and Global Annealing**

All three run on one NVIDIA Tesla V100-SXM2-32GB, with a population of
$131072$ replicas (independent replicas for Simulated Annealing), a `logT`
schedule from $T = 1.92$ to $T = 0.1$, 50 thermalization sweeps at
$T = 1.92$, and a final $T = 0$ quench of the population (single-spin flips
accepted only if they lower the energy, until none does). Each instance is
run 10 times. A run's energy is the lowest one it reaches, during the anneal
or in the quench, and its runtime includes the thermalization, the quench
and, for Global Annealing, the training of the MADE network.

| | Per temperature |
|---|---|
| Population Annealing | 10 local sweeps, multinomial resampling |
| Simulated Annealing | 10 local sweeps |
| Global Annealing | 5 global (MADE) steps, each followed by 15 local sweeps |

Global Annealing trains the MADE network for 40 epochs at the start and 1
epoch at each new temperature (batch size 256, Adam, learning rate
$10^{-3}$).

**Number of temperatures**

Each algorithm was run at several numbers of temperatures on every instance,
and each row reports, for that instance, the number of temperatures with the
lowest TTS (tuned per instance, as in the Parameters column). An instance
that an algorithm never solves reports the number of temperatures that gave
its lowest energy. Numbers of temperatures run:

| | $L = 10$ | $L = 14$ |
|---|---|---|
| Population Annealing | 20, 25, 35, 50, 80 | 375, 1000 |
| Global Annealing | 8, 10, 12, 14 (13 and 16 on 10 instances) | 30, 40, 50, 72 |
| Simulated Annealing | 20, 35, 50, 80, 120, 200 | 200, 375, 1000, 2000 |

At $L = 14$ no Simulated Annealing run reaches the reference.

**Parameters column**

| Displayed as | Raw name in `summary.csv` | Meaning |
|---|---|---|
| $\theta_g$ | `global_steps_per_temperature` (Global Annealing only) | Number of global (population-level) Monte Carlo steps performed at each temperature |
| $\theta_l$ | `MCS_per_global_steps` (Global Annealing) / `MCS_per_temperature` (Population and Simulated Annealing) | Number of local (single-spin) Monte Carlo sweeps performed between global steps (Global Annealing) or at each temperature |
| Num. temp. | `number_of_temperatures` | Number of temperatures in the annealing schedule |
| Schedule | `schedule` | Name of the temperature schedule (e.g. `logT` for logarithmic spacing) |
