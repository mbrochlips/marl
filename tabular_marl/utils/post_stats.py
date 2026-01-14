import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from post_visualizations import load_eval_returns_from_csv, get_color_from_agent_name
from scipy import stats


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


def load_agent_statistics_from_runs(experimental_data, TR = False):

    if TR:
        csv_file = "eval_returns_full.csv"
    else:
        csv_file = "eval_returns_last10.csv"

    stat_pr_rep = np.array([None,None])

    for idx, exp in enumerate(experimental_data):

        run_path = exp['run_path']
        csv_path = os.path.join(run_path, csv_file)
        all_returns = load_eval_returns_from_csv(csv_path)
        n_reps, total_eval_eps, n_agents = all_returns.shape
        
        # Get config values
        n_checkpoints = exp.get('n_checkpoints', 10)
        eval_eps_per_checkpoint = total_eval_eps // n_checkpoints
        
        # Extract only the last checkpoint returns
        # Reshape: (n_reps, n_checkpoints, eval_eps_per_checkpoint, n_agents)
        all_returns_reshaped = all_returns.reshape(n_reps, n_checkpoints, eval_eps_per_checkpoint, n_agents)
        
        if run_path[:4] == "pRand":
            mean_returns_reshaped = all_returns_reshaped[:, :, :, 1]
        else:
            mean_returns_reshaped = np.mean(all_returns_reshaped, axis=1)

        # Mean within each repetition for last checkpoint: (n_reps, n_agents)
        

        if TR:
            # as checkpoints are evenly spaced, we use np.arange(n_checkpoints) as x
            stat_pr_rep[idx] = np.sum(np.trapz(mean_returns_reshaped, dx=1, axis=1), axis=1)
        else:
            # Combined mean across both agents (if not Prand) per repetition: (n_reps,)
            rep_means = np.mean(mean_returns_reshaped, axis=1)
            stat_pr_rep[idx] = np.mean(rep_means, axis=1)

    return stat_pr_rep #list


# Figure settings
FIG_WIDTH = 5
FIG_HEIGHT = 2
FIG_ALPHA = 0.2

def hist_result_multiple_runs(experimental_data, B=10000, confidence_level=0.95, title=None, TR=False):
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
    agent_statistics = load_agent_statistics_from_runs(experimental_data,TR=TR)
    

    # Plot all bootstrap distributions and CIs in the same figure
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.5, FIG_HEIGHT * 2.5))


    for i, exp in enumerate(experimental_data):
        if 'color' in exp:
            color = exp['color']
        else:
            color = get_color_from_agent_name(exp['run_path'])
        label = exp.get('label', f'Experiment {i+1}')


        # Perform bootstrap analysis
        bootstrap_means, ci_lower, ci_upper, observed_mean, se = bootstrap_distribution(
            agent_statistics[i], B=B, confidence_level=confidence_level
        )

        # Summary statistics
        print("=" * 50)
        if TR:
            print(f'{label}: Bootstrapped TR')
        else:
            print(f'{label}: Bootstrapped AP')

        print("=" * 50)
        print(f"Observed mean ({label}): {observed_mean:.4f}")
        print(f"Bootstrap SE ({label}): {se:.4f}")
        print(f"95% CI ({label}): [{ci_lower:.4f}, {ci_upper:.4f}]")
        print("-" * 50)
        
        ax.hist(bootstrap_means, bins=50, density=True, alpha=0.4, color=color, edgecolor='white')
        ax.axvline(observed_mean, color=color, linestyle='--', linewidth=2, label=f'{label} (gns.: {observed_mean:.3f})')
        ax.axvline(ci_lower, color=color, linestyle=':', linewidth=2)
        ax.axvline(ci_upper, color=color, linestyle=':', linewidth=2)

    if TR:
        ax.set_xlabel('Total Belønning (TR)', fontsize=16)
    else:
        ax.set_xlabel('Gns. Episode Afkast (AP)', fontsize=16)

    ax.set_ylabel('Tæthed', fontsize=16)
    if title:
        ax.set_title(title, fontsize=18)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    return fig


def welch_difference_test(experiments, TR=False):

    data = load_agent_statistics_from_runs(experiments, TR=TR)
    # Welch's t-test: two-sided difference test comparing agent_1_mean vs agent_2_mean
    labels = [exp.get('label', f'Experiment {i+1}') for i, exp in enumerate(experiments)]

    statistic = "TR" if TR else "AP"
    
    # Extract the mean returns for each agent
    agent_1_stats = data[0]
    agent_2_stats = data[1]

    # Perform Welch's t-test (does not assume equal variances)
    t_statistic, p_value = stats.ttest_ind(agent_1_stats, agent_2_stats, equal_var=False)

    print("=" * 50)
    print(f"Welch's t-test: {labels[0]} {statistic} vs {labels[1]} {statistic}")
    print("=" * 50)
    # print(f"{labels[0]}: {np.mean(agent_1_stats):.4f} ± {np.std(agent_1_stats):.4f}")
    # print(f"{labels[1]}: {np.mean(agent_2_stats):.4f} ± {np.std(agent_2_stats):.4f}")
    # print(f"Difference: {np.mean(agent_1_stats) - np.mean(agent_2_stats):.4f}")
    print("-" * 50)
    print(f"t-statistic: {t_statistic:.4f}")
    print(f"p-value (two-sided): {p_value:.6f}")
    print("-" * 50)

    # Interpretation
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: SIGNIFICANT difference at α = {alpha}")
        print(f"Reject the null hypothesis: the {statistic}s are significantly different.")
    else:
        print(f"Result: NO significant difference at α = {alpha}")
        print(f"Fail to reject the null hypothesis: no significant difference between {statistic}s.")

    print("-" * 50)
    print("")