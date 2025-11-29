#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ---------- Primitive generators (do NOT do any file/path checks here) ----------

def make_uniform(n, low, high, dtype):
    """
    Uniform integers in [low, high], inclusive.

    np.random.randint(low, high) is [low, high), so we must pass high+1.
    """
    return np.random.randint(low=low, high=high + 1, size=n, dtype=dtype)


def make_gaussian(n, low, high, mu, sigma, dtype):
    """
    Gaussian-ish distribution, then rounded and clamped to [low, high].
    """
    x = np.random.normal(loc=mu, scale=sigma, size=n)
    x = np.round(x).astype(np.int64)
    x = np.clip(x, low, high)
    return x.astype(dtype)


def make_exponential_unsigned(n, low, high, lam, dtype):
    """
    Exponential(λ) on positive reals, shifted + clamped into [low, high].
    """
    x = np.random.exponential(scale=1.0 / lam, size=n)
    x = np.round(x).astype(np.int64)
    x = x + low
    x = np.clip(x, low, high)
    return x.astype(dtype)


def make_exponential_signed(n, low, high, lam, dtype):
    """
    Symmetric-ish exponential for signed: sign * Exp(λ), clamped to [low, high].
    """
    mag = np.random.exponential(scale=1.0 / lam, size=n)
    sign = np.random.choice([-1, 1], size=n)
    x = np.round(mag * sign).astype(np.int64)
    x = np.clip(x, low, high)
    return x.astype(dtype)


def sorted_from(base_array):
    return np.sort(base_array)


def reverse_from(base_array):
    return np.sort(base_array)[::-1]


def make_all_equal(n, value, dtype):
    """
    All keys identical: every element == value.
    """
    return np.full(n, value, dtype=dtype)


def make_staggered_from_uniform(n, low, high, dtype):
    """
    Generate a 'staggered' pattern:

    1) Draw uniform [low, high], inclusive.
    2) Sort ascending.
    3) Split into two halves: left (low side), right (high side).
    4) Interleave from right and left: high, low, high, low, ...

    This yields: max, min, next_max, next_min, ... pattern.
    """
    base = make_uniform(n, low, high, dtype)
    base.sort()  # ascending

    left = base[: (n + 1) // 2]   # slightly larger for odd n
    right = base[(n + 1) // 2 :]  # smaller half
    right = right[::-1]           # make it descending

    out = np.empty_like(base)

    li = 0
    ri = 0
    for i in range(n):
        if i % 2 == 0:
            # even index: take from right (high side) if available
            if ri < right.size:
                out[i] = right[ri]
                ri += 1
            else:
                out[i] = left[li]
                li += 1
        else:
            # odd index: take from left (low side) if available
            if li < left.size:
                out[i] = left[li]
                li += 1
            else:
                out[i] = right[ri]
                ri += 1

    return out


# ---------- Generation functions that assume we already decided to generate ----------

def generate_unsigned_data(n, dist_cfg, low, high):
    """
    Actually generate unsigned data (uint64) for a single (n, dist_cfg, range).

    This function does NO file existence checks.
    Caller must ensure generation is needed.
    """
    dtype = np.uint64
    t = dist_cfg["type"]

    if t == "uniform":
        data = make_uniform(n, low, high, dtype)
    elif t == "gaussian":
        mu = float(dist_cfg["mu"])
        sigma = float(dist_cfg["sigma"])
        data = make_gaussian(n, low, high, mu, sigma, dtype)
    elif t == "exponential":
        lam = float(dist_cfg["lambda"])
        data = make_exponential_unsigned(n, low, high, lam, dtype)
    elif t in ("sorted", "reverse"):
        base = make_uniform(n, low, high, dtype)
        if t == "sorted":
            data = sorted_from(base)
        else:
            data = reverse_from(base)
    elif t == "all_equal":
        value = int(dist_cfg.get("value", low))
        data = make_all_equal(n, value, dtype)
    elif t == "staggered":
        data = make_staggered_from_uniform(n, low, high, dtype)
    else:
        raise ValueError(f"Unknown unsigned dist type: {t}")

    return data


def generate_signed_data(n, dist_cfg, low, high):
    """
    Actually generate signed data (int64) for a single (n, dist_cfg, range).

    This function does NO file existence checks.
    Caller must ensure generation is needed.
    """
    dtype = np.int64
    t = dist_cfg["type"]

    if t == "uniform":
        data = make_uniform(n, low, high, dtype)
    elif t == "gaussian":
        mu = float(dist_cfg["mu"])
        sigma = float(dist_cfg["sigma"])
        data = make_gaussian(n, low, high, mu, sigma, dtype)
    elif t == "exponential":
        lam = float(dist_cfg["lambda"])
        data = make_exponential_signed(n, low, high, lam, dtype)
    elif t in ("sorted", "reverse"):
        base = make_uniform(n, low, high, dtype)
        if t == "sorted":
            data = sorted_from(base)
        else:
            data = reverse_from(base)
    elif t == "all_equal":
        value = int(dist_cfg.get("value", low))
        data = make_all_equal(n, value, dtype)
    elif t == "staggered":
        data = make_staggered_from_uniform(n, low, high, dtype)
    else:
        raise ValueError(f"Unknown signed dist type: {t}")

    return data


# ---------- Main driver ----------

def main():
    here = Path(__file__).parent
    with open(here / "config.json", "r") as f:
        cfg = json.load(f)

    seed = int(cfg["seed"])
    np.random.seed(seed)

    unsigned_out = (here / cfg["unsigned_output_dir"]).resolve()
    signed_out = (here / cfg["signed_output_dir"]).resolve()
    ensure_dir(unsigned_out)
    ensure_dir(signed_out)

    n_values = cfg["n_values"]
    unsigned_ranges = cfg["unsigned_ranges"]
    signed_ranges = cfg["signed_ranges"]
    unsigned_dists = cfg["unsigned_distributions"]
    signed_dists = cfg["signed_distributions"]

    # ---------- UNSIGNED DATASETS ----------
    seen_unsigned = set()

    for n in n_values:
        for dist in unsigned_dists:
            profile = dist["range_profile"]
            r = unsigned_ranges[profile]
            low = int(r["min"])
            high = int(r["max"])

            name = dist["name"]
            t = dist["type"]

            # Compute filename suffix "extra" WITHOUT generating any data
            if t == "gaussian":
                mu = float(dist["mu"])
                sigma = float(dist["sigma"])
                extra = f"_mu{mu}_sigma{sigma}"
            elif t == "exponential":
                lam = float(dist["lambda"])
                extra = f"_lam{lam}"
            elif t == "all_equal":
                value = int(dist.get("value", low))
                extra = f"_val{value}"
            else:
                extra = ""

            basename = f"{name}_n{n}_min{low}_max{high}_dtypeuint64_seed{seed}{extra}"

            # Avoid duplicate config entries
            if basename in seen_unsigned:
                print(f"[unsigned] Skipping duplicate config entry for {basename}")
                continue
            seen_unsigned.add(basename)

            bin_path = unsigned_out / f"{basename}.bin"
            npy_path = unsigned_out / f"{basename}.npy"
            json_path = unsigned_out / f"{basename}.json"

            # If all files exist, skip generation entirely
            if bin_path.exists() and npy_path.exists() and json_path.exists():
                print(f"[unsigned] Skipping existing {basename}")
                continue

            # Only now generate the data
            print(f"[unsigned] Generating {basename}")
            data = generate_unsigned_data(n, dist, low, high)

            meta = {
                "name": name,
                "n": int(n),
                "dtype": "uint64",
                "seed": int(seed),
                "min": int(low),
                "max": int(high),
                "range_profile": profile,
            }

            data.tofile(bin_path)
            np.save(npy_path, data)

            with open(json_path, "w") as jf:
                json.dump(meta, jf, indent=2)

    # ---------- SIGNED DATASETS ----------
    seen_signed = set()

    for n in n_values:
        for dist in signed_dists:
            profile = dist["range_profile"]
            r = signed_ranges[profile]
            low = int(r["min"])
            high = int(r["max"])

            name = dist["name"]
            t = dist["type"]

            # Compute filename suffix "extra" WITHOUT generating any data
            if t == "gaussian":
                mu = float(dist["mu"])
                sigma = float(dist["sigma"])
                extra = f"_mu{mu}_sigma{sigma}"
            elif t == "exponential":
                lam = float(dist["lambda"])
                extra = f"_lam{lam}"
            elif t == "all_equal":
                value = int(dist.get("value", low))
                extra = f"_val{value}"
            else:
                extra = ""

            basename = f"{name}_n{n}_min{low}_max{high}_dtypeint64_seed{seed}{extra}"

            if basename in seen_signed:
                print(f"[signed] Skipping duplicate config entry for {basename}")
                continue
            seen_signed.add(basename)

            bin_path = signed_out / f"{basename}.bin"
            npy_path = signed_out / f"{basename}.npy"
            json_path = signed_out / f"{basename}.json"

            if bin_path.exists() and npy_path.exists() and json_path.exists():
                print(f"[signed] Skipping existing {basename}")
                continue

            print(f"[signed] Generating {basename}")
            data = generate_signed_data(n, dist, low, high)

            meta = {
                "name": name,
                "n": int(n),
                "dtype": "int64",
                "seed": int(seed),
                "min": int(low),
                "max": int(high),
                "range_profile": profile,
            }

            data.tofile(bin_path)
            np.save(npy_path, data)

            with open(json_path, "w") as jf:
                json.dump(meta, jf, indent=2)


if __name__ == "__main__":
    main()
