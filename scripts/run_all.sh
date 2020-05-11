#!/bin/sh

timeout=3600
resv="dls2fc_1"

run_falsifier () {
    njobs=$1
    shift
    artifact=$1
    shift
    variant=$1
    shift
    options="-T $timeout -v -p 8"
    name=$artifact.falsify.$variant
    echo "sbatch --reservation=${resv} -e logs/${name}.%J.err -o logs/${name}.%J.out ./scripts/run_falsification.sh results/${name}.csv artifacts/${artifact}_benchmark/ ${options} $@"
    for (( i=1; i<=$njobs; i++ )); do
        sbatch --reservation=${resv} -e logs/${name}.%J.err -o logs/${name}.%J.out ./scripts/run_falsification.sh results/${name}.csv artifacts/${artifact}_benchmark/ ${options} $@
    done
}

run_verifier () {
    njobs=$1
    shift
    artifact=$1
    shift
    verifier=$1
    shift
    options="-T $timeout -v"
    name=$artifact.verify.$verifier
    # options="-T $timeout --eran.domain=deepzono --prop.epsilon=$extra"
    echo "sbatch --reservation=${resv} -e logs/${name}.%J.err -o logs/${name}.%J.out ./scripts/run_verification.sh results/${name}.csv artifacts/${artifact}_benchmark/ ${verifier} ${options} $@"
    for (( i=1; i<=$njobs; i++ )); do
        sbatch --reservation=${resv} -e logs/${name}.%J.err -o logs/${name}.%J.out ./scripts/run_verification.sh results/${name}.csv artifacts/${artifact}_benchmark/ ${verifier} ${options} $@
    done;
}

# ACAS
artifact="acas"
run_falsifier 24 $artifact "cleverhans_LBFGS" "--backend cleverhans.LBFGS --n_start 1 --set cleverhans.LBFGS y_target \"[[-1.0, 0.0]]\""
run_falsifier 24 $artifact "cleverhans_BasicIterativeMethod" "--backend cleverhans.BasicIterativeMethod --n_start 1"
run_falsifier 24 $artifact "cleverhans_FastGradientMethod" "--backend cleverhans.FastGradientMethod --n_start 1"
run_falsifier 24 $artifact "cleverhans_DeepFool" "--backend cleverhans.DeepFool --set cleverhans.DeepFool nb_candidate 2"

run_falsifier 24 $artifact "cleverhans_BasicIterativeMethod_eps0.5" "--backend cleverhans.BasicIterativeMethod --set cleverhans.BasicIterativeMethod eps 0.5 --n_start 1"
run_falsifier 24 $artifact "cleverhans_FastGradientMethod_eps0.5" "--backend cleverhans.FastGradientMethod --set cleverhans.FastGradientMethod eps 0.5 --n_start 1"

run_verifier 24 $artifact "neurify" "--neurify.max_thread=8"
run_verifier 24 $artifact "eran" "--eran.domain=deepzono"

# Neurify-DAVE
artifact="neurifydave"
for epsilon in 1 2 5 8 10; do
    run_falsifier 24 $artifact "eps${epsilon}.cleverhans_LBFGS" "--backend cleverhans.LBFGS --n_start 1 --set cleverhans.LBFGS y_target \"[[-1.0, 0.0]]\" --prop.epsilon=${epsilon}"
    run_falsifier 24 $artifact "eps${epsilon}.cleverhans_BasicIterativeMethod" "--backend cleverhans.BasicIterativeMethod --n_start 1 --prop.epsilon=${epsilon}"
    run_falsifier 24 $artifact "eps${epsilon}.cleverhans_FastGradientMethod" "--backend cleverhans.FastGradientMethod --n_start 1 --prop.epsilon=${epsilon}"
    run_falsifier 24 $artifact "eps${epsilon}.cleverhans_DeepFool" "--backend cleverhans.DeepFool --set cleverhans.DeepFool nb_candidate 2 --prop.epsilon=${epsilon}"

    run_falsifier 24 $artifact "eps${epsilon}.cleverhans_BasicIterativeMethod_eps0.5" "--backend cleverhans.BasicIterativeMethod --set cleverhans.BasicIterativeMethod eps 0.5 --n_start 1 --prop.epsilon=${epsilon}"
    run_falsifier 24 $artifact "eps${epsilon}.cleverhans_FastGradientMethod_eps0.5" "--backend cleverhans.FastGradientMethod --set cleverhans.FastGradientMethod eps 0.5 --n_start 1 --prop.epsilon=${epsilon}"

    run_verifier 24 $artifact "eps${epsilon}.neurify" "--neurify.max_thread=8 --prop.epsilon=${epsilon}"
    run_verifier 24 $artifact "eps${epsilon}.eran" "--eran.domain=deepzono --prop.epsilon=${epsilon}"
done
