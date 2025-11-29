#!/bin/bash
#
#SBATCH --job-name=cpu_bench_all
#SBATCH --output=logs/cpu_bench_all_%j.out
#SBATCH --error=logs/cpu_bench_all_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# SBATCH --partition=markov_cpu         
# SBATCH --account=csds438              

# ================== User-configurable defaults ==================

# Where datasets live (relative to project root)
DATASET_DIR_UINT64="${DATASET_DIR_UINT64:-datasets_unsigned}"
DATASET_DIR_INT64="${DATASET_DIR_INT64:-datasets_signed}"

# Which dtypes to run over: "uint64 int64", or override to "uint64" only, etc.
DTYPE_LIST="${DTYPE_LIST:-uint64 int64}"

# CSV metrics files (relative to project root)
CSV_PATH_UINT64="${CSV_PATH_UINT64:-logs/cpu_bench_uint64.csv}"
CSV_PATH_INT64="${CSV_PATH_INT64:-logs/cpu_bench_int64.csv}"

# Implementation list
IMPL_LIST="${IMPL_LIST:-seq omp}"

# Algorithm lists
ALGO_LIST_UINT64="${ALGO_LIST_UINT64:-radix counting bucket pigeonhole bitonic}"
# For signed int64 we currently only implement radix (via uint64 mapping)
ALGO_LIST_INT64="${ALGO_LIST_INT64:-radix}"

# OpenMP threads (also matches --cpus-per-task by default)
THREADS="${THREADS:-8}"

# Whether to verify vs std::sort (1 -> add --check, 0 -> skip)
CHECK="${CHECK:-1}"

# Number of measured repeats and warmup runs for each combination
REPEATS="${REPEATS:-3}"
WARMUP="${WARMUP:-1}"

# ================================================================

echo "SLURM job running on host: $(hostname)"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "Configured parameters:"
echo "  DATASET_DIR_UINT64 = $DATASET_DIR_UINT64"
echo "  DATASET_DIR_INT64  = $DATASET_DIR_INT64"
echo "  DTYPE_LIST         = $DTYPE_LIST"
echo "  IMPL_LIST          = $IMPL_LIST"
echo "  ALGO_LIST_UINT64   = $ALGO_LIST_UINT64"
echo "  ALGO_LIST_INT64    = $ALGO_LIST_INT64"
echo "  THREADS            = $THREADS"
echo "  CHECK              = $CHECK"
echo "  REPEATS            = $REPEATS"
echo "  WARMUP             = $WARMUP"
echo "  CSV_PATH_UINT64    = $CSV_PATH_UINT64"
echo "  CSV_PATH_INT64     = $CSV_PATH_INT64"
echo

# Go to project root (where you submit from)
cd "$SLURM_SUBMIT_DIR" || exit 1

module load Miniconda3/23.10.0-1
source ~/.bashrc
conda activate /mnt/vstor/courses/csds438/sxi219/conda_envs/sorting

# Ensure logs directory exists at project root
mkdir -p logs

# Optionally reset CSV files so this run is clean
: > "$CSV_PATH_UINT64"
: > "$CSV_PATH_INT64"



# Move into build directory where cpu_sort_bench lives
cd sorting-benchmark/build || exit 1

# Export threads for OpenMP
export OMP_NUM_THREADS="$THREADS"

echo "======== CPU BENCHMARK FULL SWEEP START ========"

for DTYPE in $DTYPE_LIST; do
  if [ "$DTYPE" = "uint64" ]; then
    DATASET_DIR="$SLURM_SUBMIT_DIR/$DATASET_DIR_UINT64"
    CSV_PATH="$SLURM_SUBMIT_DIR/$CSV_PATH_UINT64"
    ALGO_LIST="$ALGO_LIST_UINT64"
  elif [ "$DTYPE" = "int64" ]; then
    DATASET_DIR="$SLURM_SUBMIT_DIR/$DATASET_DIR_INT64"
    CSV_PATH="$SLURM_SUBMIT_DIR/$CSV_PATH_INT64"
    ALGO_LIST="$ALGO_LIST_INT64"
  else
    echo "WARNING: Unsupported DTYPE='$DTYPE' (expected uint64 or int64), skipping."
    continue
  fi

  echo
  echo "=== DTYPE=$DTYPE ==="
  echo "  DATASET_DIR = $DATASET_DIR"
  echo "  ALGO_LIST   = $ALGO_LIST"
  echo "  CSV_PATH    = $CSV_PATH"
  echo

  # Sanity check dataset directory
  if [ ! -d "$DATASET_DIR" ]; then
    echo "WARNING: Dataset directory not found: $DATASET_DIR (skipping this dtype)"
    continue
  fi

  # Loop over all .bin files in the dataset directory (non-recursive)
  find "$DATASET_DIR" -maxdepth 1 -type f -name '*.bin' | sort | while read -r BIN_PATH; do
    BASENAME=$(basename "$BIN_PATH")

    # Extract N from filename pattern ..._n<N>_min...
    N=$(echo "$BASENAME" | sed -E 's/.*_n([0-9]+)_min.*/\1/')

    if [ -z "$N" ]; then
      echo "WARNING: Could not parse N from filename '$BASENAME' (skipping file)."
      continue
    fi

    echo
    echo "---- Dataset: $BASENAME ----"
    echo "  BIN_PATH = $BIN_PATH"
    echo "  N        = $N"
    echo

    for ALGO in $ALGO_LIST; do
      for IMPL in $IMPL_LIST; do
        echo "Running combination:"
        echo "  ALGO    = $ALGO"
        echo "  IMPL    = $IMPL"
        echo "  DTYPE   = $DTYPE"
        echo "  N       = $N"
        echo "  THREADS = $THREADS"
        echo "  REPEATS = $REPEATS"
        echo "  WARMUP  = $WARMUP"
        echo

        CMD="./cpu_sort_bench \
          --algo=${ALGO} \
          --impl=${IMPL} \
          --dtype=${DTYPE} \
          --input=${BIN_PATH} \
          --n=${N} \
          --threads=${THREADS} \
          --repeats=${REPEATS} \
          --warmup=${WARMUP} \
          --csv=${CSV_PATH}"

        if [ "$CHECK" -eq 1 ]; then
          CMD="${CMD} --check"
        fi

        echo "Command: $CMD"
        time $CMD
        echo "--------------------------------------------------"
      done
    done
  done
done

echo "======== CPU BENCHMARK FULL SWEEP END =========="
