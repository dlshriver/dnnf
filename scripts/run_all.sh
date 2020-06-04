#!/bin/sh

n=6
timeout=3600
# resv="--reservation="
resv=""

run_falsifier() {
    njobs=$1
    shift
    artifact=$1
    shift
    variant=$1
    shift
    options="-T $timeout -v -p 8"
    name=$artifact.$variant
    echo "sbatch ${resv} -e logs/${name}.%J.err -o logs/${name}.%J.out ./scripts/run_falsification.sh results/${name}.csv artifacts/${artifact}_benchmark/ ${options} $@"
    for ((i = 1; i <= $njobs; i++)); do
        sbatch ${resv} -e logs/${name}.%J.err -o logs/${name}.%J.out ./scripts/run_falsification.sh results/${name}.csv artifacts/${artifact}_benchmark/ ${options} $@
    done
}

run_verifier() {
    njobs=$1
    shift
    artifact=$1
    shift
    verifier=$1
    shift
    variant=$1
    shift
    options="-T $timeout -v"
    name=$artifact.$variant
    # options="-T $timeout --eran.domain=deepzono --prop.epsilon=$extra"
    echo "sbatch ${resv} -e logs/${name}.%J.err -o logs/${name}.%J.out ./scripts/run_verification.sh results/${name}.csv artifacts/${artifact}_benchmark/ ${verifier} ${options} $@"
    for ((i = 1; i <= $njobs; i++)); do
        sbatch ${resv} -e logs/${name}.%J.err -o logs/${name}.%J.out ./scripts/run_verification.sh results/${name}.csv artifacts/${artifact}_benchmark/ ${verifier} ${options} $@
    done
}

# ACAS
artifact="acas"
run_falsifier $n $artifact "cleverhans_LBFGS" "--backend cleverhans.LBFGS --n_start 1 --set cleverhans.LBFGS y_target \"[[-1.0, 0.0]]\""
run_falsifier $n $artifact "cleverhans_BasicIterativeMethod" "--backend cleverhans.BasicIterativeMethod --n_start 1"
run_falsifier $n $artifact "cleverhans_FastGradientMethod" "--backend cleverhans.FastGradientMethod --n_start 1"
run_falsifier $n $artifact "cleverhans_DeepFool" "--backend cleverhans.DeepFool --set cleverhans.DeepFool nb_candidate 2 --n_start 1"
run_falsifier $n $artifact "cleverhans_ProjectedGradientDescent" "--backend cleverhans.ProjectedGradientDescent"
run_falsifier $n $artifact "pgd" "--backend pgd"

run_verifier $n $artifact "neurify" "neurify" "--neurify.max_thread=8"
run_verifier $n $artifact "eran" "eran" "--eran.domain=deepzono"

sleep 1

# Output Relational
artifact="outputrelational"
run_falsifier $n $artifact "cleverhans_LBFGS" "--backend cleverhans.LBFGS --n_start 1 --set cleverhans.LBFGS y_target \"[[-1.0, 0.0]]\""
run_falsifier $n $artifact "cleverhans_BasicIterativeMethod" "--backend cleverhans.BasicIterativeMethod --n_start 1"
run_falsifier $n $artifact "cleverhans_FastGradientMethod" "--backend cleverhans.FastGradientMethod --n_start 1"
run_falsifier $n $artifact "cleverhans_DeepFool" "--backend cleverhans.DeepFool --set cleverhans.DeepFool nb_candidate 2 --n_start 1"
run_falsifier $n $artifact "cleverhans_ProjectedGradientDescent" "--backend cleverhans.ProjectedGradientDescent"
run_falsifier $n $artifact "pgd" "--backend pgd"

run_verifier $n $artifact "neurify" "neurify" "--neurify.max_thread=8"
run_verifier $n $artifact "eran" "eran" "--eran.domain=deepzono"

sleep 1

# Neurify-DAVE
artifact="neurifydave"
for epsilon in 1 2 5 8 10; do
    run_falsifier $n $artifact "eps${epsilon}.cleverhans_LBFGS" "--backend cleverhans.LBFGS --n_start 1 --set cleverhans.LBFGS y_target \"[[-1.0, 0.0]]\" --prop.epsilon=${epsilon}"
    run_falsifier $n $artifact "eps${epsilon}.cleverhans_BasicIterativeMethod" "--backend cleverhans.BasicIterativeMethod --n_start 1 --prop.epsilon=${epsilon}"
    run_falsifier $n $artifact "eps${epsilon}.cleverhans_FastGradientMethod" "--backend cleverhans.FastGradientMethod --n_start 1 --prop.epsilon=${epsilon}"
    run_falsifier $n $artifact "eps${epsilon}.cleverhans_DeepFool" "--backend cleverhans.DeepFool --set cleverhans.DeepFool nb_candidate 2 --n_start 1 --prop.epsilon=${epsilon}"
    run_falsifier $n $artifact "eps${epsilon}.cleverhans_ProjectedGradientDescent" "--backend cleverhans.ProjectedGradientDescent --prop.epsilon=${epsilon}"
    run_falsifier $n $artifact "eps${epsilon}.pgd" "--backend pgd --prop.epsilon=${epsilon}"

    run_verifier $n $artifact "neurify" "eps${epsilon}.neurify" "--neurify.max_thread=8 --prop.epsilon=${epsilon}"
    run_verifier $n $artifact "eran" "eps${epsilon}.eran" "--eran.domain=deepzono --prop.epsilon=${epsilon}"

    sleep 1
done

# run tensorfuzz last
# ACAS
artifact="acas"
run_falsifier $n $artifact "tensorfuzz" "--backend tensorfuzz"
# Output Relational
artifact="outputrelational"
run_falsifier $n $artifact "tensorfuzz" "--backend tensorfuzz"
# Neurify-DAVE
artifact="neurifydave"
for epsilon in 1 2 5 8 10; do
    run_falsifier $n $artifact "eps${epsilon}.tensorfuzz" "--backend tensorfuzz --prop.epsilon=${epsilon}"
done
