#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=64G

module load gcc
. .env.d/openenv.sh
export TF_CPP_MIN_LOG_LEVEL=3

ulimit -Ss 128000

results_csv=$1
shift
artifact_dir=$1
shift

echo "python -u ./tools/run_falsification.py $results_csv $artifact_dir -M 64G -n 1 $@"
python -u ./tools/run_falsification.py $results_csv $artifact_dir -M 64G -n 1 $@
