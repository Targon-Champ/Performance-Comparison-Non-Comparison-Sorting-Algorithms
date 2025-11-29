#!/usr/bin/env python3
"""
Phase 3 GPU Sorting Analysis Script

- Loads benchmark CSV produced by gpu_sort_bench
- Computes summary statistics
- Generates multiple comparison plots
- Saves all figures under a directory (default: ./visualizations)

Usage:
    python analyze_phase3.py /path/to/phase3_gpu_results.csv [--outdir visualizations]

The CSV is expected to have columns:

dist,algo,impl,dtype,
N,threads,ms,s_per_elem,ns_per_elem,mels_per_s,
bytes,GB,GBps,
sorted,match_ref,error,
blocks,repeats,kernel_ms_avg,kernel_ms_std,total_ms_avg,total_ms_std,verified,
dataset
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------- Utilities ----------

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig, outdir: Path, name: str):
    outpath = outdir / f"{name}.png"
    fig.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[saved] {outpath}")


# ---------- Core analysis ----------

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic sanity: drop rows with missing algo or N
    df = df.dropna(subset=["algo", "N"])
    # Ensure types
    df["N"] = df["N"].astype(int)
    if "impl" not in df.columns:
        df["impl"] = "gpu"
    return df


def summarize_overall(df: pd.DataFrame):
    print("\n=== Overall Summary (all rows) ===")
    print(df[["dist", "algo", "impl", "N", "ms", "GBps"]].describe(include="all"))

    # Per-algorithm summary
    print("\n=== Per Algorithm Summary (GPU only) ===")
    gpu = df[df["impl"] == "gpu"]
    summary = (
        gpu.groupby(["algo"])
        .agg(
            runs=("ms", "count"),
            mean_ms=("ms", "mean"),
            std_ms=("ms", "std"),
            mean_GBps=("GBps", "mean"),
        )
        .sort_values("mean_ms")
    )
    print(summary)


# ---------- Visual 1: Runtime vs N per distribution ----------

def plot_runtime_vs_N_by_dist(df: pd.DataFrame, outdir: Path):
    gpu = df[df["impl"] == "gpu"].copy()

    dists = sorted(gpu["dist"].dropna().unique())
    algos = sorted(gpu["algo"].dropna().unique())

    for dist in dists:
        sub = gpu[gpu["dist"] == dist].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        for algo in algos:
            s = sub[sub["algo"] == algo]
            if s.empty:
                continue
            # Aggregate in case of duplicates (avg over repeats)
            agg = (
                s.groupby("N")
                .agg(mean_ms=("ms", "mean"), std_ms=("ms", "std"))
                .reset_index()
                .sort_values("N")
            )
            ax.errorbar(
                agg["N"],
                agg["mean_ms"],
                yerr=agg["std_ms"],
                marker="o",
                linestyle="-",
                label=algo,
                capsize=3,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("N (log scale)")
        ax.set_ylabel("Total time (ms, log scale)")
        ax.set_title(f"Runtime vs N by Algorithm — dist={dist}")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend()
        save_fig(fig, outdir, f"runtime_vs_N_dist_{dist}")


# ---------- Visual 2: Throughput vs N per distribution ----------

def plot_throughput_vs_N_by_dist(df: pd.DataFrame, outdir: Path):
    gpu = df[df["impl"] == "gpu"].copy()
    dists = sorted(gpu["dist"].dropna().unique())
    algos = sorted(gpu["algo"].dropna().unique())

    for dist in dists:
        sub = gpu[gpu["dist"] == dist].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        for algo in algos:
            s = sub[sub["algo"] == algo]
            if s.empty:
                continue
            agg = (
                s.groupby("N")
                .agg(mean_GBps=("GBps", "mean"))
                .reset_index()
                .sort_values("N")
            )
            ax.plot(
                agg["N"],
                agg["mean_GBps"],
                marker="o",
                linestyle="-",
                label=algo,
            )

        ax.set_xscale("log")
        ax.set_xlabel("N (log scale)")
        ax.set_ylabel("Throughput (GB/s)")
        ax.set_title(f"Throughput vs N by Algorithm — dist={dist}")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend()
        save_fig(fig, outdir, f"throughput_vs_N_dist_{dist}")


# ---------- Visual 3: Speedup of bucket_radix vs others ----------

def plot_bucket_radix_speedup(df: pd.DataFrame, outdir: Path):
    gpu = df[df["impl"] == "gpu"].copy()

    # We'll compare ms per (dist, N) where bucket_radix AND other algo exist
    dists = sorted(gpu["dist"].dropna().unique())
    rivals = ["radix", "bitonic"]  # compare against these

    for dist in dists:
        sub = gpu[gpu["dist"] == dist].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        for rival in rivals:
            # pivot: index=(N), columns=algo, values=mean ms
            pivot = (
                sub[sub["algo"].isin([rival, "bucket_radix"])]
                .groupby(["algo", "N"])
                .agg(mean_ms=("ms", "mean"))
                .reset_index()
                .pivot(index="N", columns="algo", values="mean_ms")
            )

            if "bucket_radix" not in pivot.columns or rival not in pivot.columns:
                continue

            # speedup = rival_ms / bucket_radix_ms
            pivot["speedup"] = pivot[rival] / pivot["bucket_radix"]
            pivot = pivot.sort_values("N")

            ax.plot(
                pivot.index,
                pivot["speedup"],
                marker="o",
                linestyle="-",
                label=f"{rival} / bucket_radix",
            )

        if not ax.lines:
            plt.close(fig)
            continue

        ax.set_xscale("log")
        ax.set_xlabel("N (log scale)")
        ax.set_ylabel("Speedup (T_rival / T_bucket_radix)")
        ax.set_title(f"Speedup vs N — bucket_radix vs rivals (dist={dist})")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend()
        save_fig(fig, outdir, f"speedup_bucket_radix_vs_rivals_{dist}")


# ---------- Visual 4: Heatmap of "best algo" per (dist, N) ----------

def plot_best_algo_heatmap(df: pd.DataFrame, outdir: Path):
    """
    For each (dist, N, impl) pick the algo with minimum mean ms.
    Restrict to impl='gpu' for this visualization.
    """
    gpu = df[df["impl"] == "gpu"].copy()
    if gpu.empty:
        return

    # For each (dist, N) compute mean ms per algo, take argmin
    agg = (
        gpu.groupby(["dist", "N", "algo"])
        .agg(mean_ms=("ms", "mean"))
        .reset_index()
    )

    # Find best algo per (dist, N)
    best = (
        agg.loc[agg.groupby(["dist", "N"])["mean_ms"].idxmin()]
        .reset_index(drop=True)
    )
    # best: columns [dist, N, algo, mean_ms]

    # Map algos to integers for heatmap
    algo_list = sorted(best["algo"].unique())
    algo_to_id = {a: i for i, a in enumerate(algo_list)}
    best["algo_id"] = best["algo"].map(algo_to_id)

    # Create pivot with rows=dist, cols=N
    dists = sorted(best["dist"].unique())
    Ns = sorted(best["N"].unique())

    matrix = np.full((len(dists), len(Ns)), np.nan)
    for i, dist in enumerate(dists):
        for j, N in enumerate(Ns):
            row = best[(best["dist"] == dist) & (best["N"] == N)]
            if not row.empty:
                matrix[i, j] = row.iloc[0]["algo_id"]

    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.cm.get_cmap("tab10", len(algo_list))

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_xticks(range(len(Ns)))
    ax.set_xticklabels([str(n) for n in Ns], rotation=45, ha="right")
    ax.set_yticks(range(len(dists)))
    ax.set_yticklabels(dists)

    ax.set_xlabel("N")
    ax.set_ylabel("Distribution")
    ax.set_title("Best GPU Algorithm per (dist, N) by runtime (lower is better)")

    # Legend mapping colors to algo names
    from matplotlib.patches import Patch
    handles = [
        Patch(color=cmap(algo_to_id[a]), label=a) for a in algo_list
    ]
    ax.legend(handles=handles, title="Best algo", bbox_to_anchor=(1.05, 1), loc="upper left")

    save_fig(fig, outdir, "best_algo_heatmap_gpu")


# ---------- Visual 5: CPU vs GPU comparison (if CPU data exists) ----------

def plot_cpu_vs_gpu_if_available(df: pd.DataFrame, outdir: Path):
    if "impl" not in df.columns:
        print("[info] No `impl` column, skipping CPU vs GPU comparison.")
        return

    impls = df["impl"].unique()
    if not (("cpu" in impls) and ("gpu" in impls)):
        print("[info] Need both impl=cpu and impl=gpu to plot CPU vs GPU; skipping.")
        return

    # For each algo+N+dist, compute speedup CPU_ms / GPU_ms based on total_ms_avg or ms
    # We'll use ms column (total time).
    cpu = df[df["impl"] == "cpu"]
    gpu = df[df["impl"] == "gpu"]

    merged = (
        cpu[["dist", "algo", "N", "ms"]]
        .rename(columns={"ms": "cpu_ms"})
        .merge(
            gpu[["dist", "algo", "N", "ms"]],
            on=["dist", "algo", "N"],
            how="inner",
        )
        .rename(columns={"ms": "gpu_ms"})
    )

    if merged.empty:
        print("[info] No overlapping CPU/GPU rows to compare; skipping.")
        return

    merged["speedup_cpu_over_gpu"] = merged["cpu_ms"] / merged["gpu_ms"]

    # Plot: for each algo, speedup vs N (maybe aggregated over dist)
    algos = sorted(merged["algo"].unique())

    for algo in algos:
        sub = merged[merged["algo"] == algo]
        if sub.empty:
            continue

        # Optionally aggregate over distributions (mean speedup)
        agg = (
            sub.groupby("N")
            .agg(mean_speedup=("speedup_cpu_over_gpu", "mean"),
                 std_speedup=("speedup_cpu_over_gpu", "std"))
            .reset_index()
            .sort_values("N")
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(
            agg["N"],
            agg["mean_speedup"],
            yerr=agg["std_speedup"],
            marker="o",
            linestyle="-",
            capsize=3,
        )
        ax.set_xscale("log")
        ax.set_xlabel("N (log scale)")
        ax.set_ylabel("Speedup (CPU_ms / GPU_ms)")
        ax.set_title(f"CPU vs GPU Speedup for algo={algo}")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        save_fig(fig, outdir, f"cpu_vs_gpu_speedup_{algo}")


# ---------- Visual 6: Algorithm ranking boxplot per distribution ----------

def plot_algo_boxplots_per_dist(df: pd.DataFrame, outdir: Path):
    gpu = df[df["impl"] == "gpu"].copy()
    if gpu.empty:
        return
    dists = sorted(gpu["dist"].dropna().unique())

    for dist in dists:
        sub = gpu[gpu["dist"] == dist]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        # We might want to normalize by N for fairness? For now, just raw ms.
        # To avoid huge range, we can filter to a subset of N or log-y.
        # Let's just use log-y.
        algo_order = sorted(sub["algo"].unique())
        data = [sub[sub["algo"] == a]["ms"].values for a in algo_order]

        ax.boxplot(data, labels=algo_order, showmeans=True)
        ax.set_yscale("log")
        ax.set_ylabel("Total time (ms, log scale)")
        ax.set_title(f"Runtime Distribution by Algorithm — dist={dist}")
        ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)

        save_fig(fig, outdir, f"algo_boxplot_ms_dist_{dist}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Phase 3 GPU analysis script.")
    parser.add_argument("csv", help="Path to phase3 GPU results CSV")
    parser.add_argument(
        "--outdir",
        default="visualizations",
        help="Directory to save visualization images (default: visualizations)",
    )
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    print(f"[info] Saving visualizations under: {outdir}")

    df = load_data(args.csv)

    # Basic summary to stdout
    summarize_overall(df)

    # Generate all plots
    plot_runtime_vs_N_by_dist(df, outdir)
    plot_throughput_vs_N_by_dist(df, outdir)
    plot_bucket_radix_speedup(df, outdir)
    plot_best_algo_heatmap(df, outdir)
    plot_cpu_vs_gpu_if_available(df, outdir)
    plot_algo_boxplots_per_dist(df, outdir)

    print("\n[done] All visualizations generated.")


if __name__ == "__main__":
    main()
