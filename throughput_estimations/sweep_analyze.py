"""Analyze sweep results and print a comparison table.

Usage:
    uv run python -m throughput_estimations.sweep_analyze
    uv run python -m throughput_estimations.sweep_analyze --sort gpu_hours
    uv run python -m throughput_estimations.sweep_analyze --pattern "sweep_*"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_results(results_dir: str, pattern: str) -> list[dict]:
    rdir = Path(results_dir)
    results = []
    for f in sorted(rdir.glob(f"{pattern}.json")):
        with open(f) as fh:
            data = json.load(fh)
        stats = data.get("stats", {})
        if "error" in stats:
            continue
        tp = stats.get("throughput", {})
        ext = stats.get("extrapolation", {})
        inp = stats.get("input_tokens", {})
        out = stats.get("output_tokens", {})
        results.append({
            "file": f.name,
            "model": data.get("model_alias", "?"),
            "mode": data.get("mode", "?"),
            "n_measured": stats.get("n_measured", 0),
            "n_failed": stats.get("n_failed", 0),
            "samples_per_sec": tp.get("samples_per_sec", 0),
            "output_tok_per_sec": tp.get("output_tok_per_sec", 0),
            "input_tok_per_sec": tp.get("input_tok_per_sec", 0),
            "avg_input_tok": inp.get("mean", 0),
            "avg_output_tok": out.get("mean", 0),
            "gpu_hours": ext.get("gpu_hours", 0),
            "gpu_hours_opt": ext.get("gpu_hours_optimistic", 0),
            "gpu_hours_pes": ext.get("gpu_hours_pessimistic", 0),
            "wall_time_s": stats.get("wall_time_s", 0),
            "max_concurrent": stats.get("max_concurrent", 0),
            "tp_size": stats.get("tp_size", 1),
            "dp_size": stats.get("dp_size", 1),
        })
    return results


def print_table(results: list[dict], sort_by: str = "gpu_hours") -> None:
    if not results:
        print("No results found.")
        return

    results.sort(key=lambda r: r.get(sort_by, float("inf")))

    # Header
    print(f"{'File':<55} {'sps':>6} {'otok/s':>8} {'avgOut':>7} {'GPU-h':>8} {'Range':>18} {'fail':>4}")
    print("-" * 115)

    best_gpu_h = results[0]["gpu_hours"] if results else 1

    for r in results:
        delta = ""
        if best_gpu_h > 0 and r["gpu_hours"] != best_gpu_h:
            pct = (r["gpu_hours"] - best_gpu_h) / best_gpu_h * 100
            delta = f" (+{pct:.0f}%)"
        range_str = f"{r['gpu_hours_opt']/1000:.1f}K-{r['gpu_hours_pes']/1000:.1f}K"
        print(
            f"{r['file']:<55} "
            f"{r['samples_per_sec']:>6.2f} "
            f"{r['output_tok_per_sec']:>8,.0f} "
            f"{r['avg_output_tok']:>7,.0f} "
            f"{r['gpu_hours']:>8,.0f}{delta:<8} "
            f"{range_str:>18} "
            f"{r['n_failed']:>4}"
        )

    print()
    print(f"Sorted by: {sort_by} | {len(results)} results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="throughput_estimations/results")
    p.add_argument("--pattern", default="generator_reflection_Qwen3.5*")
    p.add_argument("--sort", default="gpu_hours", choices=["gpu_hours", "samples_per_sec", "output_tok_per_sec", "file"])
    args = p.parse_args()

    results = load_results(args.results_dir, args.pattern)
    print_table(results, sort_by=args.sort)


if __name__ == "__main__":
    main()
