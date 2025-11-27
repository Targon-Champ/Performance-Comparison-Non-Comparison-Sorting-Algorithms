#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def generate_uniform(n: int, min_value: int, max_value: int) -> np.ndarray:
    return np.random.randint(min_value, max_value + 1, size=n, dtype=np.int64)


def generate_gaussian(
    n: int, min_value: int, max_value: int, mean: float, std: float
) -> np.ndarray:
    x = np.random.normal(loc=mean, scale=std, size=n)
    x = np.clip(x, min_value, max_value)
    return x.astype(np.int64)


def generate_exponential(
    n: int, min_value: int, max_value: int, lam: float
) -> np.ndarray:
    x = np.random.exponential(scale=1.0 / lam, size=n)
    cutoff = 7.0 / lam
    x = np.clip(x, 0.0, cutoff)
    x_norm = x / cutoff
    span = float(max_value - min_value)
    x_mapped = min_value + x_norm * span
    return x_mapped.astype(np.int64)


def generate_sorted(n: int, min_value: int, max_value: int) -> np.ndarray:
    arr = generate_uniform(n, min_value, max_value)
    arr.sort()
    return arr


def generate_reverse_sorted(n: int, min_value: int, max_value: int) -> np.ndarray:
    arr = generate_uniform(n, min_value, max_value)
    arr.sort()
    return arr[::-1]


def compute_stats(arr: np.ndarray) -> Dict[str, Any]:
    return {
        "n": int(arr.size),
        "min": int(arr.min()) if arr.size > 0 else None,
        "max": int(arr.max()) if arr.size > 0 else None,
        "mean": float(arr.mean()) if arr.size > 0 else None,
        "variance": float(arr.var()) if arr.size > 0 else None,
    }


def build_filename(
    distribution: str,
    n: int,
    min_value: int,
    max_value: int,
    seed: int,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    parts = [
        distribution,
        f"n{n}",
        f"min{min_value}",
        f"max{max_value}",
        f"seed{seed}",
    ]
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}{v}")
    return "_".join(parts)


def save_binary(path: Path, arr: np.ndarray) -> None:
    """
    Save as raw binary int64 values (little-endian).
    C++/CUDA can read this using fread or ifstream.read().
    """
    arr.astype(np.int64).tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic dataset generator (signed int64, numpy, binary, metadata)."
    )
    parser.add_argument(
        "--dist",
        required=True,
        choices=["uniform", "gaussian", "exponential", "sorted", "reverse"],
        help="Distribution type.",
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--min-value", type=int, default=0)
    parser.add_argument("--max-value", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--std", type=float, default=100.0)
    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--out-dir", type=str, default="../datasets")

    args = parser.parse_args()

    if args.max_value < args.min_value:
        raise ValueError("max-value must be >= min-value")

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    min_v = args.min_value
    max_v = args.max_value

    # ---- generation ----
    if args.dist == "uniform":
        arr = generate_uniform(args.n, min_v, max_v)
        extra = None
    elif args.dist == "gaussian":
        arr = generate_gaussian(args.n, min_v, max_v, args.mean, args.std)
        extra = {"mu": args.mean, "sigma": args.std}
    elif args.dist == "exponential":
        arr = generate_exponential(args.n, min_v, max_v, args.lam)
        extra = {"lam": args.lam}
    elif args.dist == "sorted":
        arr = generate_sorted(args.n, min_v, max_v)
        extra = None
    elif args.dist == "reverse":
        arr = generate_reverse_sorted(args.n, min_v, max_v)
        extra = None

    # ---- paths ----
    base_name = build_filename(args.dist, args.n, min_v, max_v, args.seed, extra)
    npy_path = out_dir / f"{base_name}.npy"
    bin_path = out_dir / f"{base_name}.bin"
    meta_path = out_dir / f"{base_name}.json"

    # ---- save numpy ----
    np.save(npy_path, arr)

    # ---- save raw binary ----
    save_binary(bin_path, arr)

    # ---- save JSON metadata ----
    stats = compute_stats(arr)
    meta = {
        "distribution": args.dist,
        "n": args.n,
        "min_value": min_v,
        "max_value": max_v,
        "seed": args.seed,
        "dtype": "int64",
        "stats": stats,
    }
    if extra:
        meta["params"] = extra

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("--------------------------------------------------")
    print(f"Saved NPY   : {npy_path}")
    print(f"Saved BIN   : {bin_path}")
    print(f"Saved META  : {meta_path}")
    print("--------------------------------------------------")


if __name__ == "__main__":
    main()
