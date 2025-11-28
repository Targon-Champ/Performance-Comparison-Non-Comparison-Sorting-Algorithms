```
PHASE_1_DATA_GENERATION.md
```

---

# **Phase 1 — Dataset Generation (Final Report)**

Performance Comparison of Non-Comparison Sorting Algorithms on CPU and GPU

---

# **1. Overview**

Phase 1 establishes a **complete, reproducible, HPC-ready dataset generation pipeline** to support CPU (sequential + multicore) and GPU benchmarking of five non-comparison sorting algorithms:

* **Radix Sort**
* **Counting Sort**
* **Bucket Sort**
* **Pigeonhole Sort**
* **Bitonic Sort**

The primary goal of Phase 1 is to create **high-quality datasets** that:

* Exercise all algorithmic edge-cases
* Support both **unsigned (uint64)** and **signed (int64)** integer domains
* Produce **research-grade distribution profiles**
* Include **algorithm-specific range-limited datasets**
* Are reproducible via **metadata + SLURM execution**
* Provide both `.bin` and `.npy` formats for C++/CUDA/Python

This document represents the **final Phase 1 specification and completion summary.**

---

# **2. Dataset Requirements**

The generator produces datasets that meet the following requirements:

---

## **2.1 Data Sizes**

Each distribution is generated at these sizes:

```
100
1,000
10,000
100,000
1,000,000
10,000,000
100,000,000
```

Reasons:

| Size Range | Purpose                                            |
| ---------- | -------------------------------------------------- |
| 10²–10⁴    | micro-scale behavior, OMP overhead, kernel warmups |
| 10⁵–10⁶    | cache effects, histogram scalability               |
| 10⁷        | L3 cache pressure, memory-bandwidth dominated      |
| 10⁸        | near-GPU scale, swaps become expensive             |

These sizes ensure coverage of **all performance regimes**.

---

## **2.2 Supported Distributions**

Phase 1 includes **seven** distributions:

### **Core statistical distributions**

1. **Uniform**
2. **Gaussian (Normal)**
3. **Exponential**

### **Structural distributions**

4. **Sorted**
5. **Reverse**

### **Algorithm edge-case distributions**

6. **All Keys Equal**
   *Critical for evaluating histogram-based algorithms (counting/pigeonhole/radix).*

7. **Staggered**
   Alternating pattern (e.g., `0, max, 1, max-1, ...`), producing worst-case memory access patterns.

These seven distributions enable comprehensive stress-testing across:

* entropy levels
* skewness
* locality
* monotonicity
* histogram complexity
* branch misprediction patterns
* distribution-specific failure modes of non-comparison algorithms

---

## **2.3 Signed + Unsigned Domain Support**

Phase 1 now includes **both** domains:

| Domain              | Range                      | Purpose                                              |
| ------------------- | -------------------------- | ---------------------------------------------------- |
| **unsigned uint64** | `[0, 2³²−1]` or controlled | Baseline, original non-comparison context            |
| **signed int64**    | `[-2³¹, 2³¹−1]`            | Full signed workloads for Radix-based signed sorting |

### **Signed domain justification**

Signed datasets support:

* Real-world data (financial, sensor readings, time deltas)
* Research validity for Radix signed mapping
* Signed algorithm correctness verification
* GPU based signed comparisons

All signed datasets store values as **NumPy `int64`**.

---

## **2.4 Algorithm-Limit Ranges**

Phase 1 now includes **range-restricted datasets** designed specifically for:

* Counting Sort
* Pigeonhole Sort
* Bucket Sort

Range-limited datasets (e.g., range ≤ 100k) ensure:

* algorithms do not OOM
* parallel histograms remain feasible
* comparison fairness across algorithms
* research on “operational region limits” of each algorithm

These datasets are generated with compact ranges such as:

```
[0, 999]
[0, 99,999]
[1,000,000 values with range 10,000]
```

---

# **3. Dataset Generation Implementation**

The generator script:

```
data_generation/generate_datasets.py
```

contains:

* Pure NumPy generation
* Fixed-seed reproducibility
* JSON metadata
* Range-limit detection
* Distribution-specific parameters
* **Idempotent behavior**:

  * Checks if output exists
  * **Does NOT regenerate** datasets unnecessarily
* Compatibility with older NumPy (`randint` patched)

### **Important Fix: NumPy `integers` → `randint`**

Older cluster NumPy lacked `np.random.integers`.
Phase 1 uses:

```python
np.random.randint(low, high+1, size=n, dtype=dtype)
```

to maintain **inclusive upper bounds**.

---

# **4. Distribution Generation Details**

### **4.1 Uniform**

Uses `randint(min, max+1)`.
Baseline for distribution-agnostic behavior.

---

### **4.2 Gaussian**

```
np.random.normal(mu, sigma)
clip(min, max)
```

Clipping prevents out-of-range values and produces realistic noise.

---

### **4.3 Exponential**

Generated via:

```
np.random.exponential(lam)
scaled → int64 range
```

Produces highly skewed workloads.

---

### **4.4 Sorted**

Uniform distribution → `.sort()` ascending.

Used for best-case inputs.

---

### **4.5 Reverse Sorted**

Uniform distribution → `.sort()` descending.

Used for worst-case CPU cache effects.

---

### **4.6 All Keys Equal**

Every element = constant.

Stress test for:

* OMP contention
* bucket collapse
* histogram collapse
* warp efficiency loss on GPU

---

### **4.7 Staggered**

Pattern:

```
min, max, min+1, max-1, ...
```

Designed to cause:

* worst-case locality
* pointer chasing issues
* branch misprediction amplification

---

# **5. Output File Formats**

Each dataset produces:

### **`.npy`**

Used for Python evaluation.

### **`.bin`**

Raw binary values for C++ and CUDA:

```cpp
std::ifstream f("data.bin", std::ios::binary);
v.resize(N);
f.read((char*)v.data(), N * sizeof(T));
```

### **`.json`**

Example:

```json
{
  "distribution": "uniform",
  "n": 1000000,
  "min": -2147483648,
  "max": 2147483647,
  "seed": 42,
  "dtype": "int64",
  "params": {},
  "stats": {
    "mean": -15102.3,
    "variance": 3.99e+18
  }
}
```

Includes full specification + stats.

---

# **6. Idempotent Generation**

Phase 1 now uses:

* File checks before generating:

  * If `.bin`, `.npy`, `.json` exist → **skip**
* Ensures:

  * No duplicate work
  * No accidental overwrites
  * Perfect reproducibility
  * Safe for large datasets (100M elements)

---

# **7. SLURM HPC Generation**

SLURM jobs generate the full matrix of datasets:

```
generate_datasets.slurm
generate_datasets_negative.slurm
```

Capabilities:

* Generates all 7 distributions
* Both signed and unsigned sets
* All sizes up to 100M
* Outputs all formats
* Uses cluster memory safely
* Can resume mid-generation due to idempotency

Submit via:

```
sbatch generate_datasets.slurm
```

---

# **8. Verification Procedures**

### **8.1 Validate `.npy`:**

```python
arr = np.load("file.npy")
print(arr.dtype, arr.shape)
```

### **8.2 Validate `.bin`:**

```python
raw = np.fromfile("file.bin", dtype=np.int64)
```

### **8.3 Cross-check:**

```python
assert np.array_equal(raw, arr)
```

### **8.4 Validate JSON metadata:**

```python
import json
meta = json.load(open("file.json"))
```

---

# **9. Reproducibility Guarantees**

Phase 1 now guarantees:

* Deterministic dataset generation
* Explicit seeding
* Full configuration captured in metadata
* Algorithm-limit range datasets
* Signed + unsigned domain correctness
* SLURM execution for consistency
* Bin + NPY + JSON availability
* No duplicated generation
* Environment locked via Conda YAML

This makes Phase 1 **fully publication-ready**.

---

# **10. Final Directory Structure**

```
Performance-Comparison-Non-Comparison-Sorting-Algorithms/
│
├── data_generation/
│   ├── generate_datasets.py
│   ├── generate_datasets.slurm
│   └── config.json
│
├── datasets_signed/
│   ├── gaussian_*.bin / .npy / .json
│   ├── uniform_*.bin / .npy / .json
│   ├── exponential_*.*
│   ├── sorted_*.*
│   ├── reverse_*.*
│   ├── all_equal_*.*
│   └── staggered_*.*
│
├── datasets_unsigned/
│   ├── same structure as signed domain
│
├── logs/
│
└── PHASE_1_DATA_GENERATION.md
```


