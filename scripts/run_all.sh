#!/bin/sh

timeout=3600

run_falsifier () {
    artifact=$2
    backend=$3
    extra=$4
    options="-T $timeout --prop.epsilon=$extra"
    name=$artifact$extra.falsify.$backend
    echo "sbatch --reservation=dls2fc_21 -e logs/$name.%J.err -o logs/$name.%J.out ./scripts/run_falsification.sh results/$name.csv artifacts/${artifact}_benchmark/ $options --$backend"
    for (( i=1; i<=$1; i++ )); do
        sbatch --reservation=dls2fc_21 -e logs/$name.%J.err -o logs/$name.%J.out ./scripts/run_falsification.sh results/$name.csv artifacts/${artifact}_benchmark/ $options --$backend
    done
}

run_verifier () {
    artifact=$2
    verifier=$3
    extra=$4
    name=$artifact$extra.$verifier
    options="-T $timeout --eran.domain=deepzono --prop.epsilon=$extra"
    echo "sbatch --reservation=dls2fc_21 -e logs/$name.%J.err -o logs/$name.%J.out ./scripts/run_verification.sh results/$name.csv artifacts/${artifact}_benchmark/ $verifier $options"
    for (( i=1; i<=$1; i++ )); do
        sbatch --reservation=dls2fc_21 -e logs/$name.%J.err -o logs/$name.%J.out ./scripts/run_verification.sh results/$name.csv artifacts/${artifact}_benchmark/ $verifier $options
    done;
}

# ACAS
run_falsifier 24 "acas" "pgd"
run_falsifier 24 "acas" "tensorfuzz"

run_verifier 24 "acas" "neurify"
run_verifier 24 "acas" "eran"
run_verifier 24 "acas" "planet"
run_verifier 24 "acas" "reluplex"

# ERAN-MNIST
for epsilon in 0.120 0.100 0.080 0.060 0.040 0.030 0.025 0.020 0.015 0.010 0.005; do
    run_falsifier 8 "eranmnist" "pgd" $epsilon
    run_falsifier 8 "eranmnist" "tensorfuzz" $epsilon

    run_verifier 8 "eranmnist" "neurify" $epsilon
    run_verifier 8 "eranmnist" "eran" $epsilon
done