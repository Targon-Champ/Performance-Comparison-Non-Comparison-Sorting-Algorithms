#!/bin/bash

#SBATCH --job-name=gen_signed_datasets
#SBATCH --output=logs/gen_signed_datasets_%j.out
#SBATCH --error=logs/gen_signed_datasets_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
# SBATCH --partition=markov_cpu
# SBATCH --account=csds438

# === Setup ===

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module load Miniconda3/23.10.0-1
source ~/.bashrc
conda activate /mnt/vstor/courses/csds438/sxi219/conda_envs/sorting

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
python -c "import sys; print('Python version:', sys.version)"

# === Parameters ===

# Distributions supported by generate_datasets.py
DISTS=(uniform gaussian exponential sorted reverse)

# Dataset sizes
SIZES=(100 1000 10000 100000 1000000 10000000 100000000)

# Research-standard signed 64-bit range:
#   [-2147483648, 2147483647]
MIN_VALUE=-2147483648
MAX_VALUE=2147483647

OUT_DIR="datasets_signed"

echo "Starting signed dataset generation at: $(date)"
echo "Signed range: ${MIN_VALUE} to ${MAX_VALUE}"

# === Generation Loop ===

for dist in "${DISTS[@]}"; do
  for n in "${SIZES[@]}"; do

    echo "------------------------------------------------------------"
    echo "Generating:"
    echo "  dist      = ${dist}"
    echo "  n         = ${n}"
    echo "  min_value = ${MIN_VALUE}"
    echo "  max_value = ${MAX_VALUE}"
    echo "  output    = ${OUT_DIR}"
    echo "------------------------------------------------------------"

    python data_generation/generate_datasets.py \
      --dist "${dist}" \
      --n "${n}" \
      --min-value "${MIN_VALUE}" \
      --max-value "${MAX_VALUE}" \
      --seed 42 \
      --out-dir "${OUT_DIR}"
  done
done

echo "Finished signed dataset generation at: $(date)"
