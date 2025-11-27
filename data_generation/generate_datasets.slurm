#!/bin/bash

#SBATCH --job-name=gen_all_datasets
#SBATCH --output=logs/gen_all_datasets_%j.out
#SBATCH --error=logs/gen_all_datasets_%j.err
#SBATCH --time=06:00:00
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

DISTS=(uniform gaussian exponential sorted reverse)
SIZES=(100 1000 10000 100000 1000000 10000000 100000000)

# Signed range: [-2^31, 2^31-1]
SIGNED_MIN=-2147483648
SIGNED_MAX=2147483647

# Unsigned range: [0, 2^32-1]
UNSIGNED_MAX=4294967295

SIGNED_OUT_DIR="datasets_signed"
UNSIGNED_OUT_DIR="datasets_unsigned"

echo "Starting dataset generation at: $(date)"
echo "Signed range:   ${SIGNED_MIN} to ${SIGNED_MAX}"
echo "Unsigned range: 0 to ${UNSIGNED_MAX}"

# === Signed datasets (int64) ===

for dist in "${DISTS[@]}"; do
  for n in "${SIZES[@]}"; do
    echo "------------------------------------------------------------"
    echo "SIGNED: dist=${dist}, n=${n}, min=${SIGNED_MIN}, max=${SIGNED_MAX}"
    echo "Output dir: ${SIGNED_OUT_DIR}"
    echo "------------------------------------------------------------"

    python data_generation/generate_datasets.py \
      --dist "${dist}" \
      --n "${n}" \
      --min-value "${SIGNED_MIN}" \
      --max-value "${SIGNED_MAX}" \
      --seed 42 \
      --out-dir "${SIGNED_OUT_DIR}"
  done
done

# === Unsigned datasets (uint64) ===

for dist in "${DISTS[@]}"; do
  for n in "${SIZES[@]}"; do
    echo "------------------------------------------------------------"
    echo "UNSIGNED: dist=${dist}, n=${n}, min=0, max=${UNSIGNED_MAX}"
    echo "Output dir: ${UNSIGNED_OUT_DIR}"
    echo "------------------------------------------------------------"

    python data_generation/generate_datasets.py \
      --dist "${dist}" \
      --unsigned \
      --n "${n}" \
      --max-value "${UNSIGNED_MAX}" \
      --seed 42 \
      --out-dir "${UNSIGNED_OUT_DIR}"
  done
done

echo "Finished dataset generation at: $(date)"
