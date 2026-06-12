#!/usr/bin/env python3
"""Plot multi-task model scaling from wandb training runs."""

import argparse
import os
import re
import sys
import warnings
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from jaxrl.env_names import get_environment_list

RUN_NAME_RE = re.compile(
    r"^(?P<env>.+)_(?P<arch>bro|simbaV2|xqc|flashsac)_(?P<size>\d+)(?:mln|m)_seed(?P<seed>\d+)$"
)
TASK_RETURN_COL_RE = re.compile(r"^seed(?P<task>\d+)/return$")
MAX_RETURN = 1000.0
DEFAULT_TIMESTEP = 500_000

ARCH_STYLE = {
    "bro": {"marker": "*", "color": "#1f77b4", "label": "bro"},
    "simbaV2": {"marker": "p", "color": "#2ca02c", "label": "simbaV2"},
    "xqc": {"marker": "s", "color": "#9467bd", "label": "xqc"},
    "flashsac": {"marker": "o", "color": "#7f7f7f", "label": "flashsac"},
}

X_TICKS = [0.2, 0.5, 1, 4, 16, 32, 64]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot multi-task model scaling from wandb runs."
    )
    parser.add_argument(
        "--env_names",
        default="DMC_DOGS",
        help="wandb group / benchmark name (default: DMC_DOGS)",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="wandb entity (default: WANDB_ENTITY from .env)",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="wandb project (default: WANDB_PROJECT from .env)",
    )
    parser.add_argument(
        "--output",
        default="plots/mt_scaling/multi_task_scaling.png",
        help="Output PNG path (default: plots/mt_scaling/multi_task_scaling.png)",
    )
    parser.add_argument(
        "--archs",
        nargs="+",
        default=None,
        choices=list(ARCH_STYLE.keys()),
        help="Architectures to include (default: all)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=DEFAULT_TIMESTEP,
        help=f"Training step to read metrics from (default: {DEFAULT_TIMESTEP})",
    )
    return parser.parse_args()


def parse_param_count_m(size: str) -> float:
    """Parse parameter count from run name size token (e.g. 05 -> 0.5M, 32 -> 32M)."""
    if size.startswith("0") and len(size) > 1:
        return int(size) / 10
    return float(int(size))


def parse_run_name(name: str) -> dict | None:
    match = RUN_NAME_RE.match(name)
    if not match:
        return None
    return {
        "env": match.group("env"),
        "arch": match.group("arch"),
        "param_count_m": parse_param_count_m(match.group("size")),
        "seed": int(match.group("seed")),
    }


def _row_at_timestep(history, target_timestep: int):
    if isinstance(history, list):
        if not history or "timestep" not in history[0]:
            return None
        rows = [r for r in history if r.get("timestep") == target_timestep]
        return rows[0] if rows else None

    if "timestep" not in history.columns:
        return None
    at_target = history[history["timestep"] == target_timestep]
    if at_target.empty:
        return None
    return at_target.iloc[0]


def extract_task_returns(run, target_timestep: int) -> dict[int, float] | None:
    history = run.history(pandas=True)
    if not isinstance(history, list) and history.empty:
        return None
    if isinstance(history, list) and not history:
        return None

    row = _row_at_timestep(history, target_timestep)
    if row is None:
        return None

    task_returns: dict[int, float] = {}

    if isinstance(row, dict):
        for col, value in row.items():
            match = TASK_RETURN_COL_RE.match(col)
            if match is None or value is None:
                continue
            task_returns[int(match.group("task"))] = float(value) / MAX_RETURN
    else:
        for col in history.columns:
            match = TASK_RETURN_COL_RE.match(col)
            if match is None:
                continue
            value = row[col]
            if np.isnan(value):
                continue
            task_returns[int(match.group("task"))] = float(value) / MAX_RETURN

    return task_returns or None


def fetch_run_results(
    entity: str,
    project: str,
    env_names: str,
    archs: list[str] | None,
    target_timestep: int,
):
    import wandb

    api = wandb.Api()
    runs = api.runs(
        f"{entity}/{project}",
        filters={"group": env_names},
    )

    results: list[dict] = []
    skipped = 0

    for run in runs:
        if run.state in ("crashed", "failed"):
            skipped += 1
            continue

        parsed = parse_run_name(run.name)
        if parsed is None:
            warnings.warn(f"Skipping run with unparseable name: {run.name!r}")
            skipped += 1
            continue

        if parsed["env"] != env_names:
            continue

        if archs is not None and parsed["arch"] not in archs:
            continue

        task_returns = extract_task_returns(run, target_timestep)
        if task_returns is None:
            warnings.warn(
                f"Skipping run without metrics at timestep {target_timestep}: {run.name!r}"
            )
            skipped += 1
            continue

        results.append({
            **parsed,
            "task_returns": task_returns,
            "normalized_return": float(np.mean(list(task_returns.values()))),
        })

    return results, skipped


def aggregate_results(
    results: list[dict],
    task_idx: int | None = None,
) -> dict[str, dict[float, dict[str, float]]]:
    """Group by arch -> param_count_m -> {mean, std, n_seeds}."""
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in results:
        if task_idx is None:
            value = r["normalized_return"]
        else:
            value = r["task_returns"].get(task_idx)
            if value is None:
                continue
        grouped[(r["arch"], r["param_count_m"])].append(value)

    aggregated: dict[str, dict[float, dict[str, float]]] = defaultdict(dict)
    for (arch, param_count_m), values in grouped.items():
        arr = np.array(values)
        aggregated[arch][param_count_m] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)) if len(arr) > 1 else 0.0,
            "n_seeds": len(arr),
        }
    return aggregated


def print_configuration_summary(results: list[dict]) -> None:
    """Print how many seeds are available per arch × parameter-count config."""
    grouped: dict[tuple[str, float], list[int]] = defaultdict(list)
    for r in results:
        grouped[(r["arch"], r["param_count_m"])].append(r["seed"])

    print("\nConfiguration summary (seeds per arch × parameter count):")
    for arch, param_count_m in sorted(grouped):
        seeds = sorted(grouped[(arch, param_count_m)])
        seed_list = ", ".join(f"seed{s}" for s in seeds)
        n = len(seeds)
        print(
            f"  {arch:10} {param_count_m:4g}M: "
            f"{n} seed{'s' if n != 1 else ''} ({seed_list})"
        )


def plot_output_path(base_output: str, suffix: str | None = None) -> str:
    root, ext = os.path.splitext(base_output)
    if suffix is None:
        return base_output
    return f"{root}_{suffix}{ext}"


def plot_scaling(
    aggregated: dict[str, dict[float, dict[str, float]]],
    output: str,
    title: str = "Multi-task model scaling",
    show: bool = False,
    ylim: tuple[float, float] = (0.0, 1.0),
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for arch, style in ARCH_STYLE.items():
        if arch not in aggregated:
            warnings.warn(f"No data for architecture: {arch}")
            continue

        sizes = sorted(aggregated[arch].keys())
        means = [aggregated[arch][s]["mean"] for s in sizes]
        stds = [aggregated[arch][s]["std"] for s in sizes]

        lower = [m - s for m, s in zip(means, stds)]
        upper = [m + s for m, s in zip(means, stds)]

        ax.fill_between(
            sizes,
            lower,
            upper,
            color=style["color"],
            alpha=0.2,
            linewidth=0,
        )
        ax.plot(
            sizes,
            means,
            color=style["color"],
            marker=style["marker"],
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.8,
            linewidth=2.5,
            label=style["label"],
        )

    ax.set_xscale("log")
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels([str(t) for t in X_TICKS])
    ax.set_xlim(min(X_TICKS), max(X_TICKS))

    ax.set_ylim(*ylim)
    y_min, y_max = ylim
    n_ticks = 6 if y_min == 0.0 else 4
    ax.set_yticks(np.linspace(y_min, y_max, n_ticks))

    ax.set_xlabel("Total parameter count (M)", fontsize=12)
    ax.set_ylabel("Normalized returns", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    ax.grid(True, which="major", color="#e0e0e0", linewidth=1.2)
    ax.grid(True, which="minor", color="#e0e0e0", linewidth=0.6, alpha=0.7)

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color("#333333")

    ax.tick_params(axis="both", which="major", labelsize=10, width=1.5, length=5)

    ax.legend(
        loc="lower left",
        framealpha=0.9,
        edgecolor="#333333",
        fontsize=10,
    )

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, facecolor="white")
    print(f"Saved plot to {output}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    load_dotenv()
    args = parse_args()

    entity = args.entity or os.getenv("WANDB_ENTITY")
    project = args.project or os.getenv("WANDB_PROJECT")

    if not entity or not project:
        print(
            "Error: WANDB_ENTITY and WANDB_PROJECT must be set via .env or CLI flags.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Fetching runs from {entity}/{project} "
        f"(group={args.env_names}, timestep={args.timestep})..."
    )
    results, skipped = fetch_run_results(
        entity, project, args.env_names, args.archs, args.timestep
    )

    if not results:
        print("Error: no usable runs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(results)} runs ({skipped} skipped).")
    print_configuration_summary(results)

    task_names = get_environment_list(args.env_names)

    aggregated_mean = aggregate_results(results)
    print("\nAggregated metrics (mean across tasks):")
    for arch in sorted(aggregated_mean):
        for size in sorted(aggregated_mean[arch]):
            stats = aggregated_mean[arch][size]
            print(
                f"  {arch:10} {size:4g}M: "
                f"mean={stats['mean']:.3f} std={stats['std']:.3f} "
                f"(n={stats['n_seeds']})"
            )

    plot_scaling(
        aggregated_mean,
        args.output,
        title="Multi-task model scaling (mean across tasks)",
        show=args.show,
        ylim=(0.4, 1.0),
    )

    for task_idx, task_name in enumerate(task_names):
        aggregated_task = aggregate_results(results, task_idx=task_idx)
        if not aggregated_task:
            warnings.warn(f"No data for task: {task_name}")
            continue

        print(f"\nAggregated metrics ({task_name}):")
        for arch in sorted(aggregated_task):
            for size in sorted(aggregated_task[arch]):
                stats = aggregated_task[arch][size]
                print(
                    f"  {arch:10} {size:4g}M: "
                    f"mean={stats['mean']:.3f} std={stats['std']:.3f} "
                    f"(n={stats['n_seeds']})"
                )

        plot_scaling(
            aggregated_task,
            plot_output_path(args.output, task_name),
            title=f"Multi-task model scaling ({task_name})",
            show=args.show,
        )


if __name__ == "__main__":
    main()
