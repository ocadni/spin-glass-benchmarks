for repeat in {1..1000}; do
    qsub -v N_VALUES="1000 2744" run_greedy_ea3d_dariah.pbs
done
