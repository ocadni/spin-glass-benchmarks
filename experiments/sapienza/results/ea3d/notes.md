**Parameters column (Global Annealing / Population Annealing)**

The `parameters` column reported for the Global Annealing (GA) and
Population Annealing (PA) runs above abbreviates the underlying
`summary.csv` fields as follows:

| Displayed as | Raw name in `summary.csv` | Meaning |
|---|---|---|
| $\theta_g$ | `global_steps_per_temperature` (GA only) | Number of global (population-level) Monte Carlo steps performed at each temperature |
| $\theta_l$ | `MCS_per_global_steps` (GA) / `MCS_per_temperature` (PA) | Number of local (single-spin) Monte Carlo sweeps performed between global steps (GA) or at each temperature (PA) |
| Num. temp. | `number_of_temperatures` | Number of temperatures in the annealing schedule |
| Schedule | `schedule` | Name of the temperature schedule (e.g. `logT` for logarithmic spacing) |
