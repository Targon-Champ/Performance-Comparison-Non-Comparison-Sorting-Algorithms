#!/usr/bin/env python
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def generate_uniform(n: int, min_value: int, max_value: int, dtype) -> np.ndarray:
    return np.random.randint(min_value, max_value + 1, size=n, dtype=dtype)


def generate_gaussian(n: int, min_value: int, max_value: int, mean: float, std: float, dtype) -> np.ndarray:
    x = np.random.normal(loc=mean, scale=std, size=n)
    x = np.clip(x, min_value, max_value)
    return x.astype(dtype)


def generate_exponential(n: int, min_value: int, max_value: int, lam: float, dtype) -> np.ndarray:
    x = np.random.exponential(scale=1.0 / lam, size=n)

    cutoff = 7.0 / lam
    x = np.clip(x, 0.0, cutoff)

    x_norm = x / cutoff
    span = float(max_value - min_value)
    x_mapped = min_value + x_norm * span

    return x_mapped.astype(dtype)


def generate_sorted(n: int, min_value: int, max_value: int, dtype) -> np.ndarray:
    arr = generate_uniform(n, min_value, max_value, dtype=dtype)
    arr.sort()
    return arr


def generate_reverse_sorted(n: int, min_value: int, max_value: int, dtype) -> np.ndarray:
    arr = generate_uniform(n, min_value, max_value, dtype=dtype)
    arr.sort()
    return arr[::-1]


def compute_stats(arr: np.ndarray) -> Dict[str, Any]:
    return {
        "n": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "variance": float(arr.var()),
    }


def build_filename(
        distribution: str,
        n: int,
        min_value: int,
        max_value: int,
        seed: int,
        dtype_name: str,
        extra: Optional[Dict[str, Any]] = None,
) -> str:
    parts = [
        distribution,
        f"n{n}",
        f"min{min_value}",
        f"max{max_value}",
        f"dtype{dtype_name}",
        f"seed{seed}",
    ]
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}{v}")

    return "_".join(parts)


def save_binary(path: Path, arr: np.ndarray) -> None:
    arr.tofile(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset generator with signed and unsigned int64 support.")

    parser.add_argument("--dist", required=True,
                        choices=["uniform", "gaussian", "exponential", "sorted", "reverse"])
    parser.add_argument("--n", type=int, required=True)

    # Signed range (ignored for unsigned mode)
    parser.add_argument("--min-value", type=int, default=None)
    parser.add_argument("--max-value", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--std", type=float, default=100.0)
    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--unsigned", action="store_true",
                        help="Generate unsigned uint64 datasets instead of signed int64.")

    parser.add_argument("--out-dir", type=str, default="../datasets")

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine dtype and range
    if args.unsigned:
        dtype = np.uint64
        min_v = 0
        max_v = args.max_value if args.max_value is not None else np.iinfo(np.uint64).max
        dtype_name = "uint64"
    else:
        dtype = np.int64
        min_v = args.min_value
        max_v = args.max_value
        dtype_name = "int64"

    if min_v is None or max_v is None:
        raise ValueError("min-value and max-value must be provided for signed generation.")

    if max_v < min_v:
        raise ValueError("max-value must be >= min-value.")

    # Generate dataset
    if args.dist == "uniform":
        arr = generate_uniform(args.n, min_v, max_v, dtype=dtype)
        extra = None
    elif args.dist == "gaussian":
        arr = generate_gaussian(args.n, min_v, max_v, args.mean, args.std, dtype)
        extra = {"mu": args.mean, "sigma": args.std}
    elif args.dist == "exponential":
        arr = generate_exponential(args.n, min_v, max_v, args.lam, dtype)
        extra = {"lam": args.lam}
    elif args.dist == "sorted":
        arr = generate_sorted(args.n, min_v, max_v, dtype)
        extra = None
    elif args.dist == "reverse":
        arr = generate_reverse_sorted(args.n, min_v, max_v, dtype)
        extra = None

    # Output file naming
    base_name = build_filename(args.dist, args.n, min_v, max_v, args.seed, dtype_name, extra)

    npy_path = out_dir / f"{base_name}.npy"
    bin_path = out_dir / f"{base_name}.bin"
    meta_path = out_dir / f"{base_name}.json"

    np.save(npy_path, arr)
    save_binary(bin_path, arr)

    stats = compute_stats(arr)

    metadata = {
        "distribution": args.dist,
        "n": args.n,
        "min_value": int(min_v),
        "max_value": int(max_v),
        "seed": args.seed,
        "dtype": dtype_name,
        "stats": stats,
    }
    if extra:
        metadata["params"] = extra

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved NPY:   {npy_path}")
    print(f"Saved BIN:   {bin_path}")
    print(f"Saved JSON:  {meta_path}")


if __name__ == "__main__":
    main()
