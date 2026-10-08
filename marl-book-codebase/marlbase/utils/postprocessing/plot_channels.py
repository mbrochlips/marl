# File: marlbase/plot_channels.py
from pathlib import Path
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configured for 4 channels with flexible pattern matching
CHANNEL_CONFIGS = [
    {
        "id": "ID_1",
        "label": "ID_1 (Stoch Solo, Lvl 1)",
        "color": "#E69F00",  # Orange
        "patterns": ["return_ID_1_stoch", "return_ID_1", "return_channel_0"],
    },
    {
        "id": "ID_2",
        "label": "ID_2 (Det Solo, Lvl 2)",
        "color": "#0072B2",  # Blue
        "patterns": ["return_ID_2_solo", "return_ID_2", "return_channel_1"],
    },
    {
        "id": "ID_3",
        "label": "ID_3 (Coop, Lvl 3)",
        "color": "#009E73",  # Green
        "patterns": ["return_ID_3_coop", "return_ID_3", "return_channel_2"],
    },
    {
        "id": "step_cost",
        "label": "Step Cost (Energy, Ch 3)",
        "color": "#D55E00",  # Vermillion / Red
        "patterns": ["return_step_cost", "return_step", "return_channel_3"],
    },
]


def resolve_channel_columns(columns):
    """
    Finds the exact mean and std column names in the CSV for each configured channel.
    """
    resolved = []
    cols_set = set(columns)

    for cfg in CHANNEL_CONFIGS:
        mean_col = None
        std_col = None

        for pat in cfg["patterns"]:
            # Try with 'mean_' prefix first, then raw pattern
            candidates = [f"mean_{pat}", pat]
            for c in candidates:
                if c in cols_set:
                    mean_col = c
                    break

            # Try to find corresponding std column
            std_candidates = [f"std_{pat}", f"std_{mean_col}"]
            for c in std_candidates:
                if c in cols_set:
                    std_col = c
                    break

            if mean_col is not None:
                break

        if mean_col is not None:
            resolved.append((mean_col, std_col, cfg["label"], cfg["color"]))

    return resolved


def plot_from_dataframe(df, title, save_path):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5.5), dpi=300)

    if "environment_steps" not in df.columns:
        print(f"[ERROR] 'environment_steps' column not found in CSV. Available columns: {list(df.columns)}")
        return

    steps = df["environment_steps"]
    num_agents = 2.0  # Average return per agent

    channels_to_plot = resolve_channel_columns(df.columns)

    if not channels_to_plot:
        print(f"[WARNING] No matching channel columns found in dataframe!")
        print(f"Available columns in CSV: {list(df.columns)}")
        return

    print(f"\nPlotting {len(channels_to_plot)} channels for '{title}':")
    for mean_col, std_col, label, color in channels_to_plot:
        print(f"  -> Matched {label}: '{mean_col}'" + (f" (std: '{std_col}')" if std_col else ""))
        mean = df[mean_col] / num_agents
        final_val = mean.dropna().iloc[-1] if not mean.dropna().empty else 0.0

        plt.plot(
            steps,
            mean,
            label=f"{label} (Final: {final_val:.2f})",
            color=color,
            linewidth=2.2,
        )

        if std_col and std_col in df.columns:
            std = df[std_col] / num_agents
            plt.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18)

    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Average Return per Agent", fontsize=12)
    plt.title(f"{title} - Decomposed Channel Breakdown ({len(channels_to_plot)} Channels)", fontsize=13, fontweight="bold")
    plt.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    plt.legend(frameon=True, fontsize=10, loc="best")
    plt.tight_layout()

    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file_png = out_dir / f"{title}_channels.png"
    out_file_pdf = out_dir / f"{title}_channels.pdf"

    plt.savefig(out_file_png)
    plt.savefig(out_file_pdf)
    print(f"Saved PNG to: {out_file_png}")
    print(f"Saved PDF to: {out_file_pdf}\n")
    plt.close()


@click.command()
@click.option("--source", type=click.Path(exists=True), required=True, help="Path to results.csv or run directory.")
@click.option("--save_path", type=click.Path(), default="./plots", help="Directory where plots will be saved.")
def main(source, save_path):
    source_path = Path(source)

    # 1. Direct CSV file provided
    if source_path.is_file() and source_path.suffix == ".csv":
        df = pd.read_csv(source_path)
        run_name = source_path.parent.name
        plot_from_dataframe(df, title=f"Run_{run_name}", save_path=save_path)

    # 2. Directory provided
    else:
        csv_file = source_path / "results.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            plot_from_dataframe(df, title=f"Run_{source_path.name}", save_path=save_path)
        else:
            from marlbase.utils.postprocessing.load_data import load_and_group_runs
            groups = load_and_group_runs(source_path, minimal_name=True)
            for group in groups:
                steps = group.get_metric("environment_steps").mean(axis=0)
                data = {"environment_steps": steps}
                for cfg in CHANNEL_CONFIGS:
                    for pat in cfg["patterns"]:
                        candidate = f"mean_{pat}"
                        if group.has_metric(candidate):
                            data[candidate] = group.get_metric(candidate).mean(axis=0)
                            std_cand = f"std_{pat}"
                            if group.has_metric(std_cand):
                                data[std_cand] = group.get_metric(std_cand).mean(axis=0)
                            break
                df = pd.DataFrame(data)
                plot_from_dataframe(df, title=group.name, save_path=save_path)


if __name__ == "__main__":
    main()