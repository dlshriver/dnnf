#!/bin/sh

n=6
timeout=3600
resv="dls2fc_1"

run_falsifier() {
    njobs=$1
    shift
    artifact=$1
    shift
    variant=$1
    shift
    options="-T $timeout -v -p 8"
    name=$artifact.$variant
    echo "sbatch --reservation=${resv} -e logs/${name}.%J.err -o logs/${name}.%J.out ./scripts/run_falsification.sh results/${name}.csv artifacts/${artifact}_benchmark/ ${options} $@"
    for ((i = 1; i <= $njobs; i++)); do
        sbatch --reservation=${resv} -e logs/${name}.%J.err -o logs/${name}.%J.out ./scripts/run_falsification.sh results/${name}.csv artifacts/${artifact}_benchmark/ ${options} $@
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
    echo "sbatch --reservation=${resv} -e logs/${name}.%J.err -o logs/${name}.%J.out ./scripts/run_verification.sh results/${name}.csv artifacts/${artifact}_benchmark/ ${verifier} ${options} $@"
    for ((i = 1; i <= $njobs; i++)); do
        sbatch --reservation=${resv} -e logs/${name}.%J.err -o logs/${name}.%J.out ./scripts/run_verification.sh results/${name}.csv artifacts/${artifact}_benchmark/ ${verifier} ${options} $@
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

# run_falsifier $n $artifact "cleverhans_BasicIterativeMethod_eps0.5" "--backend cleverhans.BasicIterativeMethod --set cleverhans.BasicIterativeMethod eps 0.5 --n_start 1"
# run_falsifier $n $artifact "cleverhans_FastGradientMethod_eps0.5" "--backend cleverhans.FastGradientMethod --set cleverhans.FastGradientMethod eps 0.5 --n_start 1"

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
    run_falsifier $n $artifact "cleverhans_ProjectedGradientDescent" "--backend cleverhans.ProjectedGradientDescent --prop.epsilon=${epsilon}"
    run_falsifier $n $artifact "pgd" "--backend pgd --prop.epsilon=${epsilon}"

    # run_falsifier $n $artifact "eps${epsilon}.cleverhans_BasicIterativeMethod_eps0.5" "--backend cleverhans.BasicIterativeMethod --set cleverhans.BasicIterativeMethod eps 0.5 --n_start 1 --prop.epsilon=${epsilon}"
    # run_falsifier $n $artifact "eps${epsilon}.cleverhans_FastGradientMethod_eps0.5" "--backend cleverhans.FastGradientMethod --set cleverhans.FastGradientMethod eps 0.5 --n_start 1 --prop.epsilon=${epsilon}"

    run_verifier $n $artifact "neurify" "eps${epsilon}.neurify" "--neurify.max_thread=8 --prop.epsilon=${epsilon}"
    run_verifier $n $artifact "eran" "eps${epsilon}.eran" "--eran.domain=deepzono --prop.epsilon=${epsilon}"

    sleep 1
done

# Output Relational
artifact="outputrelational"
run_falsifier $n $artifact "cleverhans_LBFGS" "--backend cleverhans.LBFGS --n_start 1 --set cleverhans.LBFGS y_target \"[[-1.0, 0.0]]\""
run_falsifier $n $artifact "cleverhans_BasicIterativeMethod" "--backend cleverhans.BasicIterativeMethod --n_start 1"
run_falsifier $n $artifact "cleverhans_FastGradientMethod" "--backend cleverhans.FastGradientMethod --n_start 1"
run_falsifier $n $artifact "cleverhans_DeepFool" "--backend cleverhans.DeepFool --set cleverhans.DeepFool nb_candidate 2 --n_start 1"
run_falsifier $n $artifact "cleverhans_ProjectedGradientDescent" "--backend cleverhans.ProjectedGradientDescent"
run_falsifier $n $artifact "pgd" "--backend pgd"

# run_falsifier $n $artifact "cleverhans_BasicIterativeMethod_eps0.5" "--backend cleverhans.BasicIterativeMethod --set cleverhans.BasicIterativeMethod eps 0.5 --n_start 1"
# run_falsifier $n $artifact "cleverhans_FastGradientMethod_eps0.5" "--backend cleverhans.FastGradientMethod --set cleverhans.FastGradientMethod eps 0.5 --n_start 1"

run_verifier $n $artifact "neurify" "neurify" "--neurify.max_thread=8"
run_verifier $n $artifact "eran" "eran" "--eran.domain=deepzono"

sleep 1

# ERAN-MNIST
artifact="eranmnist"
pf="--properties_filename=properties_3convnets.csv"
for epsilon in 0.120 0.100 0.080 0.060 0.040 0.030 0.025 0.020 0.015 0.010 0.005; do
    run_falsifier $n $artifact "eps${epsilon}.cleverhans_LBFGS" "--backend cleverhans.LBFGS --n_start 1 --set cleverhans.LBFGS y_target \"[[-1.0, 0.0]]\" --prop.epsilon=${epsilon} ${pf}"
    run_falsifier $n $artifact "eps${epsilon}.cleverhans_BasicIterativeMethod" "--backend cleverhans.BasicIterativeMethod --n_start 1 --prop.epsilon=${epsilon} ${pf}"
    run_falsifier $n $artifact "eps${epsilon}.cleverhans_FastGradientMethod" "--backend cleverhans.FastGradientMethod --n_start 1 --prop.epsilon=${epsilon} ${pf}"
    run_falsifier $n $artifact "eps${epsilon}.cleverhans_DeepFool" "--backend cleverhans.DeepFool --set cleverhans.DeepFool nb_candidate 2 --n_start 1 --prop.epsilon=${epsilon} ${pf}"
    run_falsifier $n $artifact "cleverhans_ProjectedGradientDescent" "--backend cleverhans.ProjectedGradientDescent --prop.epsilon=${epsilon} ${pf}"
    run_falsifier $n $artifact "pgd" "--backend pgd --prop.epsilon=${epsilon} ${pf}"

    # run_falsifier $n $artifact "eps${epsilon}.cleverhans_BasicIterativeMethod_eps0.5" "--backend cleverhans.BasicIterativeMethod --set cleverhans.BasicIterativeMethod eps 0.5 --n_start 1 --prop.epsilon=${epsilon} ${pf}"
    # run_falsifier $n $artifact "eps${epsilon}.cleverhans_FastGradientMethod_eps0.5" "--backend cleverhans.FastGradientMethod --set cleverhans.FastGradientMethod eps 0.5 --n_start 1 --prop.epsilon=${epsilon} ${pf}"

    run_verifier $n $artifact "neurify" "eps${epsilon}.neurify" "--neurify.max_thread=8 --prop.epsilon=${epsilon} ${pf}"
    run_verifier $n $artifact "eran" "eps${epsilon}.eran" "--eran.domain=deepzono --prop.epsilon=${epsilon} ${pf}"

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
# ERAN-MNIST
artifact="eranmnist"
pf="--properties_filename=properties_3convnets.csv"
for epsilon in 0.120 0.100 0.080 0.060 0.040 0.030 0.025 0.020 0.015 0.010 0.005; do
    run_falsifier $n $artifact "eps${epsilon}.tensorfuzz" "--backend tensorfuzz --prop.epsilon=${epsilon} ${pf}"
done
