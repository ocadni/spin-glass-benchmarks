Backup of the global-annealing (GA) code and GA launch scripts, taken on 2026-10-06
before speeding up GA's global (MADE) step.  State: fused-kernel local sweeps in place
(fused_sweep.py), float64 quench, learning-rate option, batch-size fix.
To restore a file, copy it back to the same relative path under annealing_solvers/.

Also included (unchanged files that solver.py imports), so that this copy runs on its own:
  python Modern/backup_ga_before_speedup_2026-10-06/Modern/optimization/solver.py INSTANCE ga ...
