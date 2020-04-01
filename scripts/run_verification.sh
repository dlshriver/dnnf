#!/bin/bash
#SBATCH --cpus-per-task=8

module load gcc
. .env/openenv.sh

ulimit -Ss 128000

export GRB_LICENSE_FILE=/u/dls2fc/gurobi_lic/$(hostname)/gurobi.lic

results_csv=$1; shift
artifact_dir=$1; shift
verifier=$1; shift

echo "python -u ./tools/run_verification.py $results_csv $artifact_dif $verifier -M 64G -n 1 $@"
python -u ./tools/run_verification.py $results_csv $artifact_dif $verifier -M 64G -n 1 $@
