#!/bin/sh

timeout=3600

run_falsifier () {
    artifact=$2
    extraname=$3
    options="-p 8 -T $timeout"
    echo "sbatch --reservation=dls2fc_21 -e logs/$artifact$extraname.falsify.%J.err -o logs/$artifact$extraname.falsify.%J.out ./scripts/run_falsification.sh tmp/results/$artifact$extraname.falsify.csv artifacts/$artifact_benchmark/ $options"
    for (( i=1; i<=$1; i++ )); do
        sbatch --reservation=dls2fc_21 -e logs/$artifact$extraname.falsify.%J.err -o logs/$artifact$extraname.falsify.%J.out ./scripts/run_falsification.sh tmp/results/$artifact$extraname.falsify.csv artifacts/$artifact_benchmark/ $options
    done
}

run_verifier () {
    artifact=$2
    verifier=$3
    extraname=$4
    options="-T $timeout --eran.domain=deepzono --neurify.max_thread=8"
    echo "sbatch --reservation=dls2fc_21 -e logs/$artifact$extraname.$verifier.%J.err -o logs/$artifact$extraname.$verifier.%J.out ./scripts/run_verification.sh results/$artifact$extraname.$verifier.csv artifacts/${artifact}_benchmark/ $verifier $options"
    for (( i=1; i<=$1; i++ )); do
        sbatch --reservation=dls2fc_21 -e logs/$artifact$extraname.$verifier.%J.err -o logs/$artifact$extraname.$verifier.%J.out ./scripts/run_verification.sh results/$artifact$extraname.$verifier.csv artifacts/${artifact}_benchmark/ $verifier $options
    done;
}

# ACAS
run_falsifier 24 "acas"

for verifier in neurify eran planet reluplex; do
    run_verifier 24 "acas" $verifier
done

# ERAN-MNIST
for epsilon in 0.120 0.100 0.080 0.060 0.040 0.030 0.025 0.020 0.015 0.010 0.005; do
    run_falsifier 24 "eranmnist" $epsilon

    for verifier in neurify eran; do
        run_verifier 24 "eranmnist" $verifier $epsilon
    done
done