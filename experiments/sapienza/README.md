# sapienza

This folder contains the solvers, run scripts, and results submitted to the benchmark under the uploader name `sapienza`. Results from this folder appear on the [Results](https://ocadni.github.io/spin-glass-benchmarks/results.html) page with `sapienza` in the Uploader column. The list of all algorithms in the benchmark is on the [Algorithms](https://ocadni.github.io/spin-glass-benchmarks/algorithms.html) page.

| Folder | Contents |
|---|---|
| [`code/greedy_solvers/`](code/greedy_solvers/) | Zero-temperature descents: [Greedy](#greedy), [Reluctant](#reluctant), [Random](#random) (`greedy.c`) |
| [`code/annealing_solvers/Modern/`](code/annealing_solvers/Modern/) | Annealing methods: [Simulated annealing](#simulated-annealing), [Population annealing](#population-annealing), [Global annealing](#global-annealing) (entry point: `optimization/solver.py`) |
| [`results/<family>/`](results/) | `summary.csv` with the reported runs and `notes.md` with the protocol for each problem family |

A worked example that runs these solvers and reproduces some of the published results is in [`example_notebook/tutorial.ipynb`](../../example_notebook/tutorial.ipynb).

The notation (energy $H$, local field $h_i^{\mathrm{loc}}$, single-flip energy change $\Delta E_i$, sweep) follows the [Notation and conventions](https://ocadni.github.io/spin-glass-benchmarks/algorithms.html#sec-notation) section of the Algorithms page.

**Contents**

1. [Zero-temperature descents](#zero-temperature-descents): [Greedy](#greedy), [Reluctant](#reluctant), [Random](#random)
2. [Annealing methods](#annealing-methods): [Simulated annealing](#simulated-annealing), [Population annealing](#population-annealing), [Global annealing](#global-annealing)
3. [References](#references)

---

## Zero-temperature descents

### General formulation

A zero-temperature descent starts from a uniformly random configuration and repeatedly flips one unsatisfied spin, so that the energy strictly decreases at every step. It terminates when no unsatisfied spin remains, that is, at a local minimum. Descents in this class differ only in their **selection rule**, which determines which unsatisfied spin is flipped when several are available.

The selection rule determines the statistics of the local minima reached: different rules converge to minima at different typical energies, after a different number of flips [[Parisi 2003](#references); [Contucci et al. 2005](#references); [Erba et al. 2024](#references)]. This folder implements three rules:

| Rule | Spin flipped at each step | Character |
|---|---|---|
| Greedy | The one with the most negative $\Delta E_i$ | Steepest descent |
| Reluctant | The one with the least negative $\Delta E_i$ | Minimal-decrease descent |
| Random | An unsatisfied spin chosen uniformly at random | Unbiased descent |

### Greedy

**Definition.** At each step, the greedy rule flips the spin that produces the largest energy decrease:

```math
i^{\star} = \arg\min_{i\,:\,\Delta E_i < 0} \Delta E_i .
```

This is steepest descent on the discrete energy landscape. Ties are broken uniformly at random.

**Rationale.** Greedy descent converges in the smallest number of flips of the three rules, typically of order $N$. Because the largest energy decreases are taken first, large groups of spins are fixed early in the descent, and the final configuration remains close to the basin of the initial condition. On the SK model, greedy descent converges to energies substantially above the ground-state energy [[Parisi 2003](#references); [Erba et al. 2024](#references)].

### Reluctant

**Definition.** At each step, the reluctant rule flips the unsatisfied spin that produces the smallest energy decrease:

```math
i^{\star} = \arg\max_{i\,:\,\Delta E_i < 0} \Delta E_i .
```

Ties are broken uniformly at random.

**Rationale.** Reluctant descent applies the smallest available change at each step, which perturbs the configuration as little as possible and delays the fixing of large groups of spins. On the SK model, it reaches significantly lower energies than greedy descent, at the cost of a substantially larger number of flips before convergence [[Parisi 2003](#references); [Contucci et al. 2005](#references); [Erba et al. 2024](#references)]. The comparison between the greedy and the reluctant rules shows that the rate of energy decrease along a trajectory does not determine the depth of the minimum it reaches.

### Random

**Definition.** At each step, the random rule selects an unsatisfied spin uniformly at random and flips it.

**Rationale.** Random descent introduces no bias toward large or small energy decreases. It provides a reference against which the effect of the biased selection rules of greedy and reluctant descent can be assessed.

### Implementation

The three descents are implemented in a single C program, [`code/greedy_solvers/greedy.c`](code/greedy_solvers/greedy.c), running on the CPU. Local fields are updated incrementally: after spin $k$ is flipped, only the fields of the neighbours of $k$ are recomputed. The set of unsatisfied spins is stored in a data structure suited to each rule:

- for **greedy** and **reluctant** descent, a binary heap ordered by $\Delta E_i$ (ascending for greedy, descending for reluctant), so that the next spin to be flipped is always at the root;
- for **random** descent, an array supporting constant-time insertion, deletion, and uniform sampling.

A spin is considered unsatisfied only if $\Delta E_i < -10^{-6}$, to avoid flips driven by floating-point rounding on degenerate moves. Each run consists of a single descent from an independent random configuration, carried out until convergence. Runs are repeated many times per instance with independent seeds. The mean number of flips per run is reported in the Average steps column of the [Results](https://ocadni.github.io/spin-glass-benchmarks/results.html) page.

### Role in the benchmark

Zero-temperature descents serve as a baseline. They have no tunable parameters, so their results do not depend on configuration choices. Their low cost allows thousands of independent runs per instance, which gives well-determined estimates of success probability and time to solution. Since every run terminates in a local minimum, their results also characterize the energy of typical local minima on each problem family.

---

## Annealing methods

### General formulation

#### Boltzmann sampling and the Metropolis rule

Annealing methods sample configurations from the Boltzmann distribution

```math
P_\beta(\mathbf{s}) = \frac{e^{-\beta H(\mathbf{s})}}{Z(\beta)},
\qquad \beta = 1/T ,
```

at a sequence of decreasing temperatures. At very high temperatures, $P_\beta$ is close to uniform and configuration space can be explored efficiently. As $T \to 0$, $P_\beta$ concentrates on the ground states. The standard local dynamics for sampling $P_\beta$ is the Metropolis rule [[Metropolis et al. 1953](#references)]: a proposed flip of spin $i$ is accepted with probability

```math
A_i = \min\!\left(1,\, e^{-\beta\, \Delta E_i}\right).
```

Moves that lower the energy are always accepted, and moves that raise it are accepted with a probability that decreases exponentially with $\beta\,\Delta E_i$. At low temperature, leaving a local minimum requires a sequence of such unfavourable moves, and the relaxation time of local dynamics increases rapidly. Annealing methods address this by lowering the temperature gradually, so that the configurations at each temperature serve as a starting point for the next.

#### Temperature schedules

An annealing run visits a sequence of $K$ temperatures $T_1 > T_2 > \dots > T_K$ between $T_{\mathrm{start}}$ and $T_{\mathrm{end}}$. Common choices are schedules linear in $T$, linear in $\beta$, and geometric in $T$:

```math
T_k = T_{\mathrm{start}}
      \left( \frac{T_{\mathrm{end}}}{T_{\mathrm{start}}} \right)^{\frac{k-1}{K-1}},
\qquad k = 1, \dots, K .
```

A geometric schedule allocates relatively more temperatures to the low-temperature region, where relaxation is slowest.

#### Populations of replicas

Many annealing methods evolve a **population** of $R$ configurations (replicas) simultaneously. Methods differ in whether and how replicas interact: they may evolve independently, be selected according to their energies, or share information through a model fitted to the population. A population also maps naturally onto parallel hardware, since the same update can be applied to all replicas in a single batched operation.

### Simulated annealing

**Definition.** Simulated annealing (SA) [[Kirkpatrick et al. 1983](#references)] applies [Metropolis dynamics](#boltzmann-sampling-and-the-metropolis-rule) while lowering the temperature along a [schedule](#temperature-schedules):

1. initialize the configurations at random and thermalize them at $T_1$;
2. for $k = 2, \dots, K$, perform $\theta_l$ Metropolis sweeps at temperature $T_k$;
3. return the lowest-energy configuration encountered.

**Rationale.** SA is motivated by the physical annealing of materials, in which slow cooling allows a system to reach a low-energy ordered state, while rapid cooling produces a disordered glass. At high temperature, the dynamics crosses energy barriers easily. As the temperature decreases, it is progressively confined to low-energy regions. If the cooling is slow enough for the system to remain close to equilibrium at every temperature, the run ends near the ground state.

In spin glasses, equilibration times grow rapidly as the temperature decreases. A schedule of practical length cannot maintain equilibrium down to low temperature, and each trajectory eventually becomes confined to a single region of the landscape. The outcome of SA therefore depends on the schedule length, and a large number of independent trajectories is generally needed to find the ground state with high probability.

**Configuration of the reported runs.** The reported SA runs evolve $R = 10^5$ independent trajectories in parallel on a GPU. The trajectories do not interact; the population provides a large number of independent attempts within a single run, and the energy of the run is the lowest energy found across the population. Each temperature is followed by a single sweep ($\theta_l = 1$), with $K = 100$ temperatures for $N \le 800$ and $K = 200$ for $N \ge 1000$. With these settings, the total number of sweeps matches that of population annealing, which uses ten times fewer temperatures and ten sweeps per temperature. The details shared with the other annealing implementations are given in [Implementation details](#implementation-details).

### Population annealing

**Definition.** Population annealing (PA) [[Hukushima and Iba 2003](#references); [Machta 2010](#references)] extends SA with a resampling step between temperatures. When the inverse temperature is increased from $\beta$ to $\beta'$, each replica $a$, of energy $E_a$, is assigned the normalized weight

```math
w_a = \frac{e^{-(\beta' - \beta) E_a}}{\sum_{b=1}^{R} e^{-(\beta' - \beta) E_b}} ,
```

and a new population of $R$ replicas is drawn from the current one, each replica being selected with probability $w_a$ (multinomial resampling). Low-energy replicas tend to be duplicated and high-energy replicas tend to be removed. The algorithm is:

1. initialize the configurations at random and thermalize them at $T_1$;
2. for $k = 2, \dots, K$, resample the population from $T_{k-1}$ to $T_k$, then perform $\theta_l$ Metropolis sweeps at $T_k$;
3. return the lowest-energy configuration encountered.

**Rationale.** The weights $w_a$ are the ratio of the Boltzmann distributions at the two temperatures. Resampling therefore converts a population in equilibrium at $\beta$ into an approximately equilibrated population at $\beta'$, and the subsequent Metropolis sweeps reduce the residual deviation from equilibrium. From the optimization point of view, resampling reallocates computational effort from replicas confined to high-energy regions toward replicas that have reached lower energies. In SA, by contrast, a trajectory confined to a poor region continues to consume resources until the end of the run.

The main limitation of PA is the loss of diversity. Since resampling duplicates replicas, the population may become dominated by the descendants of a small number of ancestors occupying the same region of configuration space. If that region does not contain the ground state, local Metropolis dynamics is generally too slow to leave it. PA is therefore most effective when the population is large relative to the number of relevant low-energy regions.

**Configuration of the reported runs.** The reported PA runs use $\theta_l = 10$ sweeps per temperature and multinomial resampling. On SK, they use $R = 10^5$ replicas, with $K = 10$ temperatures for $N \le 800$ and $K = 20$ for $N \ge 1000$. On EA3D, the number of temperatures varies by instance and is listed in the Parameters column of the [Results](https://ocadni.github.io/spin-glass-benchmarks/results.html) tables. The details shared with the other annealing implementations are given in [Implementation details](#implementation-details).

### Global annealing

**Definition.** Global annealing (GA) is the machine-learning-enhanced Monte Carlo method of [[Del Bono et al. 2026](#references)] (see also [[Del Bono et al. 2025](#references)]). It retains the population and the local Metropolis sweeps of SA, and adds **global moves**: proposals that update all spins of a replica simultaneously, drawn from a generative model fitted to the current population.

The generative model is an autoregressive distribution, which factorizes the probability of a configuration into a product of conditional probabilities:

```math
q_\theta(\mathbf{s}) = \prod_{i=1}^{N} q_\theta(s_i \mid s_1, \dots, s_{i-1}) .
```

The implementation uses a shallow MADE network [[Germain et al. 2015](#references); [Biazzo 2023](#references)], consisting of a single masked linear layer, in which each conditional takes the form

```math
q_\theta(s_i \mid s_{<i})
= \frac{e^{\,s_i \phi_i}}{2\cosh \phi_i},
\qquad
\phi_i = \sum_{j<i} W_{ij} s_j \;(+\, b_i),
```

where $W$ is strictly lower triangular and the biases $b_i$ are included only for instances with external fields. Each conditional is the distribution of a single spin in an effective field $\phi_i$ generated by the preceding spins, a structure that mirrors the Boltzmann distribution of a pairwise Ising model. The model has two properties required by the method:

- **exact sampling**: configurations are generated sequentially, drawing $s_1$, then $s_2$ conditioned on $s_1$, and so on, without a Markov chain;
- **exact likelihood**: $q_\theta(\mathbf{s})$ can be evaluated for any configuration in a single forward pass.

A global move proposes, for each replica, a configuration $\mathbf{s}' \sim q_\theta$ drawn independently of the current configuration $\mathbf{s}$, and accepts it with the Metropolis–Hastings probability

```math
A(\mathbf{s} \to \mathbf{s}')
= \min\!\left(1,\;
  \frac{e^{-\beta E(\mathbf{s}')}\, q_\theta(\mathbf{s})}
       {e^{-\beta E(\mathbf{s})}\, q_\theta(\mathbf{s}')}
  \right).
```

The ratio $q_\theta(\mathbf{s})/q_\theta(\mathbf{s}')$ corrects for the bias of the proposal distribution, so that the dynamics satisfies detailed balance with respect to $P_\beta$ regardless of the accuracy of the model. The accuracy of the model affects only the acceptance rate. The algorithm is:

1. initialize the configurations at random and thermalize them at $T_1$;
2. train the model on the thermalized population by maximum likelihood;
3. for $k = 2, \dots, K$, repeat $\theta_g$ times a global move followed by $\theta_l$ Metropolis sweeps at $T_k$, then retrain the model on the updated population (except at the last temperature);
4. return the lowest-energy configuration encountered.

**Rationale.** SA and PA rely exclusively on single-spin moves. Moving between distinct low-energy regions with such moves requires a long sequence of intermediate configurations of higher energy, which becomes exponentially improbable at low temperature. Global moves are not subject to this constraint. The model, fitted to the population, approximates the distribution of low-energy configurations, and can propose a configuration in a different region of configuration space in a single step. The proposal is likely to be accepted if its energy is low.

The temperature schedule serves two purposes in GA. At each temperature, the model is trained on samples obtained at a slightly higher temperature, where equilibration is easier, and is then used to generate proposals at the next one. Since consecutive temperatures are close, a short retraining is sufficient to keep the model accurate. The local sweeps between global moves refine the proposed configurations at short length scales, which the model captures less accurately.

The cost of a global move is significant. Generating one configuration requires $N$ sequential steps, each involving a partial matrix–vector product, for a total cost of order $N^2$. This is comparable to one sweep on SK and considerably larger than one sweep on EA3D. Model training also contributes to the running time and is included in the reported times. Whether the larger moves compensate for their cost is one of the questions the benchmark is designed to answer, and the answer may differ between dense and sparse problem families [[Del Bono et al. 2026](#references); [Ciarella et al. 2023](#references)].

**Configuration of the reported runs.** The reported GA runs use $\theta_g = 5$ global moves per temperature, each followed by $\theta_l = 15$ Metropolis sweeps. On SK, they use $R = 10^5$ replicas, with $K = 10$ temperatures for $N \le 800$ and $K = 20$ for $N \ge 1000$. On EA3D, $K$ varies by instance and is listed in the Parameters column of the [Results](https://ocadni.github.io/spin-glass-benchmarks/results.html) tables. The model is trained with the Adam optimizer (learning rate $10^{-3}$, batch size 256) by minimizing the binary cross-entropy, which equals the negative log-likelihood of the population. Initial training runs for up to 40 epochs with learning-rate decay and early stopping; retraining at each subsequent temperature runs for 1 epoch. The details shared with the other annealing implementations are given in [Implementation details](#implementation-details).

### Implementation details

The SA, PA, and GA implementations in [`code/annealing_solvers/Modern/optimization/`](code/annealing_solvers/Modern/optimization/) share the conventions below.

- **Schedule.** All reported runs use a geometric (`logT`) schedule. On SK, $T_{\mathrm{start}} = 1.5$ and $T_{\mathrm{end}} = 0.1$, and the population is thermalized with 50 sweeps at $T_{\mathrm{start}}$ before annealing begins.
- **Update order.** The order of single-spin updates within a sweep is determined from the coupling graph. If the graph is bipartite, as is the case for the EA3D lattices with even linear size, the spins are divided into two sublattices with no internal couplings, and each sublattice is updated in parallel (checkerboard update). Otherwise, as on the complete SK graph, spins are updated sequentially in a random order drawn anew at each sweep.
- **Final quench.** After the last temperature, each replica is quenched at $T = 0$: single-spin flips are accepted only if they lower the energy, and sweeps are repeated until a full sweep produces no flips. This removes residual thermal excitations, so that every replica ends in a local minimum.
- **Reported energy.** The energy of a run is the lowest energy reached by any replica, either during the anneal or after the final quench. The reported running time includes the quench and, for GA, model training.

---

## References

- Biazzo, I. (2023). The autoregressive neural network architecture of the Boltzmann distribution of pairwise interacting spins systems. *Communications Physics* 6, 296. [doi:10.1038/s42005-023-01416-5](https://doi.org/10.1038/s42005-023-01416-5)
- Ciarella, S., Trinquier, J., Weigt, M., and Zamponi, F. (2023). Machine-learning-assisted Monte Carlo fails at sampling computationally hard problems. *Machine Learning: Science and Technology* 4, 010501. [doi:10.1088/2632-2153/acbe91](https://doi.org/10.1088/2632-2153/acbe91)
- Contucci, P., Giardinà, C., Giberti, C., Unguendoli, F., and Vernia, C. (2005). Interpolating greedy and reluctant algorithms. *Optimization Methods and Software* 20, 509–514. [doi:10.1080/10556780500140177](https://doi.org/10.1080/10556780500140177)
- Del Bono, L. M., Ricci-Tersenghi, F., and Zamponi, F. (2025). Performance of machine-learning-assisted Monte Carlo in sampling from simple statistical physics models. *Physical Review E* 112, 045307. [doi:10.1103/s1rm-29zx](https://doi.org/10.1103/s1rm-29zx)
- Del Bono, L. M., Ricci-Tersenghi, F., and Zamponi, F. (2026). Demonstrating real advantage of machine-learning-enhanced Monte Carlo for combinatorial optimization. *Proceedings of the National Academy of Sciences* 123, e2534768123. [doi:10.1073/pnas.2534768123](https://doi.org/10.1073/pnas.2534768123)
- Erba, V., Behrens, F., Krzakala, F., and Zdeborová, L. (2024). Quenches in the Sherrington–Kirkpatrick model. *Journal of Statistical Mechanics: Theory and Experiment* 2024, 083302. [doi:10.1088/1742-5468/ad685a](https://doi.org/10.1088/1742-5468/ad685a)
- Germain, M., Gregor, K., Murray, I., and Larochelle, H. (2015). MADE: Masked autoencoder for distribution estimation. *Proceedings of the 32nd International Conference on Machine Learning*, PMLR 37, 881–889. [arXiv:1502.03509](https://arxiv.org/abs/1502.03509)
- Hukushima, K., and Iba, Y. (2003). Population annealing and its application to a spin glass. *AIP Conference Proceedings* 690, 200–206. [doi:10.1063/1.1632123](https://doi.org/10.1063/1.1632123)
- Kirkpatrick, S., Gelatt, C. D., and Vecchi, M. P. (1983). Optimization by simulated annealing. *Science* 220, 671–680. [doi:10.1126/science.220.4598.671](https://doi.org/10.1126/science.220.4598.671)
- Machta, J. (2010). Population annealing with weighted averages: A Monte Carlo method for rough free-energy landscapes. *Physical Review E* 82, 026704. [doi:10.1103/PhysRevE.82.026704](https://doi.org/10.1103/PhysRevE.82.026704)
- Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., and Teller, E. (1953). Equation of state calculations by fast computing machines. *The Journal of Chemical Physics* 21, 1087–1092. [doi:10.1063/1.1699114](https://doi.org/10.1063/1.1699114)
- Parisi, G. (2003). On the statistical properties of the large time zero temperature dynamics of the SK model. *Fractals* 11 (supp01), 161–171. [arXiv:cond-mat/9501045](https://arxiv.org/abs/cond-mat/9501045)
