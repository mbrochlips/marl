# File: marlbase/plot_channels.py
from pathlib import Path
import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CHANNELS = [
    ("mean_return_ID_1_stoch", "std_return_ID_1_stoch", "ID_1 (Stoch Solo, Lvl 1)", "#E69F00"),
    ("mean_return_ID_2_solo",  "std_return_ID_2_solo",  "ID_2 (Det Solo, Lvl 2)",   "#0072B2"),
    ("mean_return_ID_3_coop",  "std_return_ID_3_coop",  "ID_3 (Coop, Lvl 3)",       "#009E73"),
]


def plot_from_dataframe(df, title, save_path):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5.5), dpi=300)

    steps = df["environment_steps"]
    num_agents = 2.0  # Average return across both agents

    for mean_col, std_col, label, color in CHANNELS:
        if mean_col in df.columns:
            mean = df[mean_col] / num_agents
            plt.plot(steps, mean, label=f"{label} (Final: {mean.iloc[-1]:.2f})", color=color, linewidth=2.2)

            if std_col in df.columns:
                std = df[std_col] / num_agents
                plt.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18)

    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel("Average Return per Agent", fontsize=12)
    plt.title(f"{title} - Decomposed Channel Breakdown", fontsize=14, fontweight="bold")
    plt.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    plt.legend(frameon=True, fontsize=11, loc="best")
    plt.tight_layout()

    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file_png = out_dir / f"{title}_channels.png"
    out_file_pdf = out_dir / f"{title}_channels.pdf"

    plt.savefig(out_file_png)
    plt.savefig(out_file_pdf)
    print(f"Saved PNG to: {out_file_png}")
    print(f"Saved PDF to: {out_file_pdf}")


@click.command()
@click.option("--source", type=click.Path(exists=True), required=True, help="Path to results.csv or a run directory.")
@click.option("--save_path", type=click.Path(), default="./plots", help="Directory where plots will be saved.")
def main(source, save_path):
    source_path = Path(source)

    # 1. Direct CSV file provided
    if source_path.is_file() and source_path.suffix == ".csv":
        df = pd.read_csv(source_path)
        run_name = source_path.parent.name
        plot_from_dataframe(df, title=f"Run_{run_name}", save_path=save_path)

    # 2. Directory provided (use marlbase grouping utility)
    else:
        csv_file = source_path / "results.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            plot_from_dataframe(df, title=f"Run_{source_path.name}", save_path=save_path)
        else:
            from marlbase.utils.postprocessing.load_data import load_and_group_runs
            groups = load_and_group_runs(source_path, minimal_name=True)
            for group in groups:
                # Build dummy DataFrame from group metrics
                steps = group.get_metric("environment_steps").mean(axis=0)
                data = {"environment_steps": steps}
                for mean_col, std_col, _, _ in CHANNELS:
                    if group.has_metric(mean_col):
                        data[mean_col] = group.get_metric(mean_col).mean(axis=0)
                        data[std_col] = group.get_metric(mean_col).std(axis=0)
                df = pd.DataFrame(data)
                plot_from_dataframe(df, title=group.name, save_path=save_path)


if __name__ == "__main__":
    main()