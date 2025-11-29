#!/bin/bash
#
#SBATCH --job-name=phase3_gpu_bench
#SBATCH --output=logs/phase3_gpu_bench_%j.out
#SBATCH --error=logs/phase3_gpu_bench_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu2080
#SBATCH --account=csds438

echo "=== Loading Modules ==="
module purge
module load Miniconda3/23.10.0
module load CUDA/12.2.0
conda activate sorting

echo "=== Environment Ready ==="

CSV="/mnt/vstor/courses/csds438/sxi219/phase3_gpu_resultsV4.csv"
BENCH="/mnt/vstor/courses/csds438/sxi219/Performance-Comparison-Non-Comparison-Sorting-Algorithms/sorting-benchmark/build/gpu_sort_bench"

DATASET_ROOT="/mnt/vstor/courses/csds438/sxi219/Performance-Comparison-Non-Comparison-Sorting-Algorithms/datasets_unsigned"

# Only the fast GPU algorithms
ALGOS=("radix" "bitonic" "bucket_radix")

# Distributions to sweep
DISTS=("uniform" "gaussian" "exponential" "staggered" "all_equal" "reverse" "sorted")

# Problem sizes
SIZES=(10000 100000 1000000 10000000 100000000)

# Start with a fresh CSV file
rm -f "$CSV"

echo "=== Starting Phase 3 GPU Sweep ==="

for dist in "${DISTS[@]}"; do
  for N in "${SIZES[@]}"; do

    DATA=$(ls $DATASET_ROOT/${dist}_n${N}_* 2>/dev/null | head -n 1)

    if [[ -z "$DATA" ]]; then
      echo "[WARN] Missing dataset: ${dist} n=${N}"
      continue
    fi

    echo "-----------------------------------------------------------"
    echo "Running: dist=${dist}, N=${N}"
    echo "Dataset: $DATA"
    echo "-----------------------------------------------------------"

    for algo in "${ALGOS[@]}"; do
      echo ">> Algorithm: $algo"
      $BENCH "$DATA" "$algo" 0 0 1 3 "$CSV"
    done

  done
done
python /mnt/vstor/courses/csds438/sxi219/Performance-Comparison-Non-Comparison-Sorting-Algorithms/analyze_phase3.py $CSV --outdir visualizations_phase3V4
echo "=== Phase 3 GPU Sweep Complete ==="
