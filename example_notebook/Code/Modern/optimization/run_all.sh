for repeat in {1..100}; do
    for n in 50 100; do
        bash run_greedy_sk.sh $n
    done
done