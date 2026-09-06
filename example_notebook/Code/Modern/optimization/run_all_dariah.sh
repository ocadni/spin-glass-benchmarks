for repeat in {1..1000}; do
    qsub -v N_VALUES="50 100 200 300 400 600 800 1000 1200 1400 1600 2000" run_greedy_sk_dariah.pbs
done