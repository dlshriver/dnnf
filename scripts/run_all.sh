#!/bin/sh

timeout=3600

run_falsifier () {
    artifact=$2
    extra=$3
    options="-T $timeout --prop.epsilon=$extra"
    echo "sbatch --reservation=dls2fc_21 -e logs/$artifact$extra.falsify.%J.err -o logs/$artifact$extra.falsify.%J.out ./scripts/run_falsification.sh results/$artifact$extra.falsify.csv artifacts/${artifact}_benchmark/ $options"
    for (( i=1; i<=$1; i++ )); do
        sbatch --reservation=dls2fc_21 -e logs/$artifact$extra.falsify.%J.err -o logs/$artifact$extra.falsify.%J.out ./scripts/run_falsification.sh results/$artifact$extra.falsify.csv artifacts/${artifact}_benchmark/ $options
    done
}

run_verifier () {
    artifact=$2
    verifier=$3
    extra=$4
    options="-T $timeout --eran.domain=deepzono --prop.epsilon=$extra"
    echo "sbatch --reservation=dls2fc_21 -e logs/$artifact$extra.$verifier.%J.err -o logs/$artifact$extra.$verifier.%J.out ./scripts/run_verification.sh results/$artifact$extra.$verifier.csv artifacts/${artifact}_benchmark/ $verifier $options"
    for (( i=1; i<=$1; i++ )); do
        sbatch --reservation=dls2fc_21 -e logs/$artifact$extra.$verifier.%J.err -o logs/$artifact$extra.$verifier.%J.out ./scripts/run_verification.sh results/$artifact$extra.$verifier.csv artifacts/${artifact}_benchmark/ $verifier $options
    done;
}

# ACAS
# run_falsifier 24 "acas"

# for verifier in neurify eran planet reluplex; do
#     run_verifier 24 "acas" $verifier
# done
run_verifier 24 "acas" "eran"

# ERAN-MNIST
for epsilon in 0.120 0.100 0.080 0.060 0.040 0.030 0.025 0.020 0.015 0.010 0.005; do
    run_falsifier 8 "eranmnist" $epsilon

    for verifier in neurify eran; do
        run_verifier 8 "eranmnist" $verifier $epsilon
    done
done