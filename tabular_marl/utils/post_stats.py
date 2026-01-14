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

        expected_total_eval_eps =  exp.get("total_eval_eps", 100)

        # If to many eval_eps
        if total_eval_eps != expected_total_eval_eps:
            stepsize = total_eval_eps // expected_total_eval_eps
            all_returns = all_returns[:, ::stepsize, :]
        
        print("30x100xagents", all_returns.shape)
 
        # Collapse the agent dimension first
        if run_path.startswith("pRand"):
            # Assuming Agent 1 (index 1) is the learner against random
            data_agents = all_returns[:, :, 1] 
        else:
            # Collaborative/Self-play: Mean across agents
            data_agents = np.mean(all_returns, axis=2)

        
      

        # --- 2. Calculate Metric (TR vs AP) ---
        
        if TR:
            n_checkpoints = exp.get('n_checkpoints', 10)
            
            eps_per_checkpoint = total_eval_eps // n_checkpoints
            
            # Reshape to separate Time (Checkpoints) from Variance (Episodes per checkpoint)
            # Shape: (n_reps, n_checkpoints, eps_per_checkpoint)
            data_reshaped = data_agents.reshape(n_reps, n_checkpoints, eps_per_checkpoint)
            
            # Average over the episodes within a specific checkpoint
            # Shape: (n_reps, n_checkpoints)
            learning_curve = np.mean(data_reshaped, axis=2)
            
            # Calculate Area Under Curve (Total Reward)
            # dx=1 assumes checkpoints are equidistant
            metric_values = np.trapz(learning_curve, dx=1, axis=1)

        else:
            # --- AVERAGE PERFORMANCE (Robust Final Estimate) ---
            # The file contains a block of episodes representing the final state.
            # We average over ALL episodes in this file for each repetition.
            # Shape: (n_reps,)
            metric_values = np.mean(data_agents, axis=1)

        stat_pr_rep[idx] = metric_values

    return stat_pr_rep


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
    print(f"p-value (two-sided): {p_value} or {p_value:.4f}")
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