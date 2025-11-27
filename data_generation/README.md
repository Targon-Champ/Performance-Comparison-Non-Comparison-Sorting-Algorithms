
---

# Phase 1 — Dataset Generation

Performance Comparison of Non-Comparison Sorting Algorithms on CPU and GPU

---

## 1. Overview

Phase 1 establishes a complete, reproducible, research-grade dataset generation pipeline for benchmarking five non-comparison sorting algorithms across CPU, multicore, and GPU architectures.

The goals of this phase are:

* Generate datasets across a wide range of sizes
* Include multiple statistical distributions
* Use signed 64-bit integer domains
* Output formats compatible with Python, C++, and CUDA
* Preserve full metadata and reproducibility
* Provide HPC-scale generation via SLURM

This document formally concludes Phase 1.

---

# 2. Dataset Requirements

The dataset generator must satisfy the following:

### 2.1 Dataset Sizes

The following sizes are generated:

```
100
1,000
10,000
100,000
1,000,000
10,000,000
100,000,000
```

These sizes exercise different performance regimes and memory behaviors of CPU/GPU algorithms.

### 2.2 Statistical Distributions

Five distributions are included:

1. Uniform
2. Gaussian (Normal)
3. Exponential
4. Sorted (ascending)
5. Reverse sorted (descending)

These cover:

* Random workloads
* Realistic noise patterns
* Skewed workloads
* Best-case and worst-case inputs

### 2.3 Signed 64-bit Integer Support

All values are generated in the signed 32-bit representable range:

```
[-2,147,483,648 , +2,147,483,647]
```

Reasons:

* Real-world workloads frequently include negative values
* Validates Radix Sort implementation for signed domains
* Enhances generalizability
* Ensures experiments are publication-grade

Internally, all values are stored as `int64` (`numpy.int64`), ensuring cross-language compatibility.

---

# 3. Distributions in Detail

### 3.1 Uniform Distribution

Samples integers uniformly in `[min_value, max_value]`.

Applications:

* Baseline performance
* Balanced random workloads

### 3.2 Gaussian Distribution

Samples from `N(mean, std)` and clips into the valid range.

Applications:

* Physics simulations
* Sensor noise
* Real-world measurement data

### 3.3 Exponential Distribution

Samples from an exponential distribution on `[0, ∞)` then normalizes and maps to the full signed range.

Applications:

* Rare-event heavy workloads
* Queueing and time-interval models
* Skewed data stress-testing

### 3.4 Sorted Distribution

Uniformly generated values followed by ascending sort.

Used for:

* Best-case scenarios
* Adaptive sorting algorithms

### 3.5 Reverse Sorted Distribution

Uniform values followed by descending sort.

Used for:

* Worst-case scenarios
* Cache line behavior testing
* Branch misprediction studies

---

# 4. Generator Implementation

The dataset generator is implemented in:

```
data_generation/generate_datasets.py
```

Key features:

* Signed 64-bit integers
* Uniform, Gaussian, Exponential, Sorted, Reverse
* Raw binary `.bin` output for C++/CUDA
* Numpy `.npy` output for Python
* Metadata `.json` for reproducibility
* Deterministic via user-provided `--seed`
* Configurable via CLI

---

# 5. Command Line Interface

Usage:

```
python generate_datasets.py \
    --dist <uniform|gaussian|exponential|sorted|reverse> \
    --n <size> \
    --min-value <int> \
    --max-value <int> \
    --seed <int> \
    --out-dir <path>
```

Optional parameters:

```
--mean <float>    (Gaussian)
--std <float>     (Gaussian)
--lam <float>     (Exponential)
```

---

# 6. Output File Formats

Each dataset produces three files:

### 6.1 Numpy File (.npy)

* Stores data in Numpy format
* Used by Python analysis tools
* Preserves `int64` dtype exactly

Example filename:

```
uniform_n100000_min-2147483648_max2147483647_seed42.npy
```

### 6.2 Raw Binary File (.bin)

* Raw contiguous 64-bit integers
* Loadable with `fread`, `ifstream.read`, or CUDA device reads
* No headers or metadata

C++ example:

```cpp
std::ifstream f("file.bin", std::ios::binary);
std::vector<int64_t> v(N);
f.read((char*)v.data(), N * sizeof(int64_t));
```

### 6.3 Metadata File (.json)

Contains distribution parameters, statistics, and reproducibility metadata.

Example:

```json
{
  "distribution": "uniform",
  "n": 1000000,
  "min_value": -2147483648,
  "max_value": 2147483647,
  "seed": 42,
  "dtype": "int64",
  "stats": {
    "n": 1000000,
    "min": -2147479183,
    "max": 2147483645,
    "mean": -15102.3,
    "variance": 3.99e+18
  }
}
```

---

# 7. HPC SLURM Large-Scale Dataset Generation

Large-scale generation is performed using the SLURM script:

```
generate_datasets_negative.slurm
```

This script:

* Loads Miniconda
* Activates the controlled environment
* Generates all dataset sizes
* Iterates over all distribution types
* Writes all `.npy`, `.bin`, `.json` files

### SLURM Generation Summary

* Distributions: 5
* Sizes: 7
* Total datasets: 35
* Each dataset produces:

  * `.npy`
  * `.bin`
  * `.json`

Submit via:

```
sbatch generate_datasets_negative.slurm
```

---

# 8. Verification

### 8.1 Validate Numpy File

```python
import numpy as np
arr = np.load("file.npy")
print(arr.shape, arr.dtype)
```

### 8.2 Validate Binary File

```python
bin_arr = np.fromfile("file.bin", dtype=np.int64)
```

### 8.3 Ensure Exact Match

```python
assert np.array_equal(arr, bin_arr)
```

### 8.4 Check Metadata

```python
import json
meta = json.load(open("file.json"))
print(meta)
```

---

# 9. Reproducibility Guarantees

This pipeline provides:

* Fixed seed control
* Full metadata recording
* Deterministic dataset generation
* Raw binary for C++/CUDA reproducibility
* Numpy format for Python reproducibility
* Conda-locked environment preventing version drift
* SLURM scripts ensuring consistent HPC execution

This ensures long-term validity and publishable reliability.

---

# 10. Directory Structure

A typical repository after Phase 1:

```
Performance-Comparison-Non-Comparison-Sorting-Algorithms/
│
├── data_generation/
│   └── generate_datasets.py
│
├── datasets_signed/
│   ├── *.npy
│   ├── *.bin
│   └── *.json
│
├── slurm/
│   └── generate_datasets_negative.slurm
│
└── PHASE_1_DATA_GENERATION.md
```

---

# Phase 1 Completed

The dataset generation pipeline is now:

* Fully implemented
* HPC-ready
* Reproducible
* Compatible with Python, C++, and CUDA
* Capable of generating large, diverse, signed 64-bit datasets
* Complete with documentation, metadata, and SLURM automation


