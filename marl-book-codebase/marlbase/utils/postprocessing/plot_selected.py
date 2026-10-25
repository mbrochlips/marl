import os
import ast
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Headless mode for HPC compute nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Figure settings
FIG_WIDTH = 5
FIG_HEIGHT = 2
FIG_ALPHA = 0.2
METRIC = "mean_episode_returns"

# The exact runs to compare
RUNS = [
    {
        "label": "MAPPO-TCOT",
        "color": "#388E3C",  # Green
        "path": "outputs/utils.custom_envs:3Foods-Penalty-6x6-v0/mappo_tcot/2026-10-24_20-21-03/results.csv",
    },
    {
        "label": "MAPPO-OT",
        "color": "#7B1FA2",  # Purple
        "path": "outputs/utils.custom_envs:3Foods-Penalty-6x6-v0/mappo_ot/2026-10-25_11-15-30/results.csv",
    },
    {
        "label": "OptiMAPPO",
        "color": "#F57C00",  # Orange
        "path": "outputs/utils.custom_envs:3Foods-Scalar-Penalty-6x6-v0/optimappo/2026-10-24_20-11-02/results.csv",
    },
    {
        "label": "MAPPO",
        "color": "#D32F2F",  # Red
        "path": "outputs/utils.custom_envs:3Foods-Scalar-Penalty-6x6-v0/mappo/2026-10-25_10-59-49/results.csv",
    },
]


def extract_env_name_from_runs(runs):
    for run in runs:
        parts = Path(run["path"]).parts
        if "outputs" in parts:
            idx = parts.index("outputs")
            raw_env = parts[idx + 1]
        else:
            raw_env = Path(run["path"]).parents[2].name

        normalized_env = raw_env.replace("-Scalar", "")
        clean_name = normalized_env.split(":")[-1] if ":" in normalized_env else normalized_env
        return clean_name
    return "UnknownEnv"


def parse_column_values(df, col_name):
    """Safely extracts and parses numerical or stringified array data from a column."""
    vals = []
    for val in df[col_name]:
        if isinstance(val, str):
            try:
                val = ast.literal_eval(val)
            except Exception:
                val = float(val)
        vals.append(val)
    arr = np.array(vals, dtype=float)

    # If multi-agent dimension exists (shape: checkpoints, agents), average across agents
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    # If multiple repetitions exist (shape: reps, checkpoints), average across reps
    if arr.ndim == 2:
        arr = arr.mean(axis=0)

    return arr


def load_returns(csv_path, metric_col=METRIC):
    df = pd.read_csv(csv_path)
    # Strip any leading/trailing spaces from column headers
    df.columns = df.columns.str.strip()

    # 1. Load Mean metric
    if metric_col not in df.columns:
        raise KeyError(f"Metric '{metric_col}' not found in {csv_path}. Columns: {list(df.columns)}")
    means = parse_column_values(df, metric_col)

    # 2. Derive and load matching Standard Deviation metric
    # e.g., 'mean_episode_returns' -> 'std_episode_returns'
    if metric_col.startswith("mean_"):
        std_col = metric_col.replace("mean_", "std_", 1)
    else:
        std_col = f"std_{metric_col}"

    if std_col in df.columns:
        stds = parse_column_values(df, std_col)
    else:
        print(f"Warning: '{std_col}' not found in {csv_path}. Shaded band will be zero.")
        stds = np.zeros_like(means)

    # 3. Retrieve environment steps
    if "environment_steps" in df.columns:
        x = df["environment_steps"].values[:len(means)]
    else:
        x = np.arange(1, len(means) + 1)

    return x, means, stds


def main():
    env_name = extract_env_name_from_runs(RUNS)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.5, FIG_HEIGHT * 2.5))

    for run in RUNS:
        if not os.path.exists(run["path"]):
            print(f"Warning: File not found: {run['path']}")
            continue

        x, means, stds = load_returns(run["path"], metric_col=METRIC)

        # Plot mean curve
        ax.plot(
            x,
            means,
            label=run["label"],
            color=run["color"],
            linewidth=2,
            marker="o",
            markersize=4,
        )

        # Plot standard deviation band
        if np.any(stds > 0):
            ax.fill_between(
                x,
                means - stds,
                means + stds,
                alpha=FIG_ALPHA,
                color=run["color"],
            )

    # English labels and formatting
    ax.set_xlabel("Environment Steps", fontsize=16)
    ax.set_ylabel("Average Return", fontsize=16)
    ax.set_title(f"{env_name} Comparison", fontsize=18)
    ax.legend(loc="best", fontsize=12)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Destination directory
    out_dir = Path("utils/postprocessing/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_env_filename = env_name.replace(":", "_").replace("/", "_")
    out_path = out_dir / f"{safe_env_filename}_{timestamp}.pdf"

    plt.savefig(out_path)
    print(f"Plot saved successfully to: {out_path}")


if __name__ == "__main__":
    main()