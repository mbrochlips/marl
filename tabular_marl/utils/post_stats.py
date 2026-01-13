import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import os
from glob import glob
from post_visualizations import load_eval_returns_from_csv


def _resolve_run_path(run_path):
    """
    Resolve a run path to the full path. If just a folder name is provided,
    automatically search in output/Final/ and output/Multiple/.
    
    :param run_path: Path to run folder (can be just folder name or full path)
    :return: Resolved full path
    """
    # If path already exists as-is, return it
    if os.path.exists(run_path):
        return run_path
    
    # If path contains directory separators, it's already a path (but doesn't exist)
    if '/' in run_path or '\\' in run_path:
        return run_path
    
    # Otherwise, it's just a folder name - try to find it in common locations
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get to tabular_marl/
    tabular_marl_dir = os.path.dirname(script_dir)
    
    # Try output/Final/ first (for 30 reps runs)
    base_paths = [
        os.path.join(tabular_marl_dir, 'output', 'Final'),
        os.path.join(tabular_marl_dir, 'output', 'Multiple'),
    ]
    
    for base_path in base_paths:
        full_path = os.path.join(base_path, run_path)
        if os.path.exists(full_path):
            return full_path
    
    # If not found, return the original path (will cause error later)
    return run_path


def bootstrap_distribution(data, B=10000, confidence_level=0.95, b=10):
    """
    Compute bootstrap distribution for given data.
    
    :param data: List of values or list of lists (for nested bootstrap)
    :param B: Number of bootstrap samples (default 10000)
    :param confidence_level: Confidence level for CI (default 0.95)
    :param b: Number of bootstrap samples per data point when data is nested (default 10)
    :return: bootstrap_means, ci_lower, ci_upper, observed_mean, se
    """
    if isinstance(data[0], list):
        n = len(data[0])
        data = np.array([np.mean(np.random.choice(d, size=n, replace=True)) for d in data for _ in range(b)])
        N = len(data)
        observed_mean = np.mean(data)
    else:
        data = np.array(data)
        N = len(data)
        observed_mean = np.mean(data)

    bootstrap_means = np.array([
        np.mean(np.random.choice(data, size=N, replace=True))
        for _ in range(B)
    ])
    
    # CI
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    # Bootstrap standard error
    se = np.std(bootstrap_means)
    
    return bootstrap_means, ci_lower, ci_upper, observed_mean, se


def load_agent_statistics_from_runs(experiment_data):
    for idx, exp in enumerate(experiment_data):
        
        # Load data from eval_returns_last10.csv
        run_path = exp['run_path']
        csv_path = os.path.join(run_path, 'eval_returns_last10.csv')
        all_returns = load_eval_returns_from_csv(csv_path)
        n_reps, total_eval_eps, n_agents = all_returns.shape
        
        # Get config values
        n_checkpoints = exp.get('n_checkpoints', 10)
        eval_eps_per_checkpoint = total_eval_eps // n_checkpoints
        
        # Extract only the last checkpoint returns
        # Reshape: (n_reps, n_checkpoints, eval_eps_per_checkpoint, n_agents)
        all_returns_reshaped = all_returns.reshape(n_reps, n_checkpoints, eval_eps_per_checkpoint, n_agents)
        
        # Get last checkpoint: (n_reps, eval_eps_per_checkpoint, n_agents)
        last_checkpoint_returns = all_returns_reshaped[:, -1, :, :]
        if run_path[:4] == "pRand":
            last_checkpoint_returns = all_returns_reshaped[1, :, :, :]
        
        # Mean within each repetition for last checkpoint: (n_reps, n_agents)
        rep_means = np.mean(last_checkpoint_returns, axis=1)
        
        # Combined mean across both agents per repetition: (n_reps,)
        combined_rep_means = np.mean(rep_means, axis=1)
        
        # Overall statistics
        overall_mean = np.mean(combined_rep_means)
        overall_std = np.std(combined_rep_means)
        
        return combined_rep_means


def hist_result_multiple_runs(experiment_data, 
                               B=10000, confidence_level=0.95, title=None):
    """
    Perform bootstrap analysis and create histograms for agent statistics from multiple runs.
    Similar to hist_result but works with multiple run folders.
    
    :param run_paths: List of run folder paths (or folder names), or a single folder path/name containing multiple runs.
                      If just a folder name is provided (e.g., "JalAM_vs_JalAM_env-mc_30reps_300eps_50epL_13jan2026"),
                      it will automatically search in output/Final/ and output/Multiple/.
    :param agent_idx: Which agent to analyze (0 for agent 1, 1 for agent 2)
    :param csv_filename: Name of the CSV file to load (default 'eval_returns.csv')
    :param B: Number of bootstrap samples (default 10000)
    :param confidence_level: Confidence level for CI (default 0.95)
    :param title: Optional title for the plot
    :return: matplotlib figure
    """
    # Load agent statistics from all runs
    agent_statistics = load_agent_statistics_from_runs(experiment_data)
    
    if len(agent_statistics) == 0:
        raise ValueError("No data loaded from the provided run paths")
    
    # Perform bootstrap analysis
    bootstrap_means, ci_lower, ci_upper, observed_mean, se = bootstrap_distribution(
        agent_statistics, B=B, confidence_level=confidence_level
    )

    # Summary statistics
    print(f"Observed mean: {observed_mean:.4f}")
    print(f"Bootstrap SE: {se:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Number of repetitions: {len(agent_statistics)}")

    # Plot bootstrap distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histograms
    axes[0].hist(bootstrap_means, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    axes[0].axvline(observed_mean, color='red', linestyle='--', linewidth=2, label=f'Observed mean: {observed_mean:.3f}')
    axes[0].axvline(ci_lower, color='orange', linestyle=':', linewidth=2, label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
    axes[0].axvline(ci_upper, color='orange', linestyle=':', linewidth=2)
    axes[0].set_xlabel('Bootstrap Mean')
    axes[0].set_ylabel('Density')
    agent_label = f'Agent {agent_idx + 1}'
    axes[0].set_title(f'Bootstrap Distribution of {agent_label} Mean Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Comparison - original distribution
    axes[1].hist(agent_statistics, bins=20, density=True, alpha=0.7, color='forestgreen', edgecolor='white')
    axes[1].axvline(observed_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {observed_mean:.3f}')
    axes[1].set_xlabel('Mean Return per Repetition')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Original Distribution of {agent_label} Mean Returns')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    plt.show()
    
    return fig

