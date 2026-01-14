import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from post_visualizations import load_eval_returns_from_csv, get_color_from_agent_name
from scipy import stats

def bootstrap_distribution(data, B=10000, confidence_level=0.95, b=10):
    """
    Compute bootstrap distribution for given data.
    """
    # Handle nested lists/arrays
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, np.ndarray)):
        flat_samples = []
        for d in data:
            if len(d) > 0:
                flat_samples.extend(np.random.choice(d, size=len(d), replace=True))
        data = np.array(flat_samples)
    
    data = np.array(data)
    N = len(data)
    
    if N == 0:
        return np.zeros(B), 0, 0, 0, 0
        
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


def load_agent_statistics_from_runs(experimental_data, TR=False):
    """
    Loads statistics.
    
    Logic:
    - If TR (Total Reward): Uses 'eval_returns_full.csv'.
      It reshapes the flat episodes into (Checkpoints, Eps_per_chk).
      It averages Eps_per_chk to get a curve point.
      It calculates np.trapz (Area Under Curve) across Checkpoints.
      
    - If AP (Average Performance): Uses 'eval_returns_last10.csv'.
      This file contains dense evaluations at the end of training.
      It simply averages ALL episodes in this file to get the robust final performance estimate.
    """

    if TR:
        csv_file = "eval_returns_full.csv"
    else:
        csv_file = "eval_returns_last10.csv"

    stat_pr_rep = []

    for idx, exp in enumerate(experimental_data):
        run_path = exp['run_path']
        csv_path = os.path.join(run_path, csv_file)
        
        
        # Shape: (n_reps, total_eval_eps, n_agents)
        all_returns = load_eval_returns_from_csv(csv_path)

        # Check if error of to many evals:
        #if len(all_returns[1]) != 100:
        print(len(all_returns[1]))
        

        n_reps, total_eval_eps, n_agents = all_returns.shape
        
 
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

        stat_pr_rep.append(metric_values)

    return stat_pr_rep


# Figure settings
FIG_WIDTH = 5
FIG_HEIGHT = 2
FIG_ALPHA = 0.2

def hist_result_multiple_runs(experimental_data, B=10000, confidence_level=0.95, title=None, TR=False):
    agent_statistics = load_agent_statistics_from_runs(experimental_data, TR=TR)
    
    # Check if we have valid data before plotting
    if not any(len(x) > 0 for x in agent_statistics):
        print("No valid data found to plot.")
        return None

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.5, FIG_HEIGHT * 2.5))

    for i, exp in enumerate(experimental_data):
        if 'color' in exp:
            color = exp['color']
        else:
            color = get_color_from_agent_name(exp['run_path'])
        label = exp.get('label', f'Experiment {i+1}')

        if len(agent_statistics) <= i or len(agent_statistics[i]) == 0:
            continue

        # Perform bootstrap analysis
        bootstrap_means, ci_lower, ci_upper, observed_mean, se = bootstrap_distribution(
            agent_statistics[i], B=B, confidence_level=confidence_level
        )

        print("=" * 50)
        print(f'{label}: Bootstrapped {"TR" if TR else "AP"}')
        print("=" * 50)
        print(f"Observed mean: {observed_mean:.4f}")
        print(f"Bootstrap SE: {se:.4f}")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print("-" * 50)
        
        # Plotting
        # Handle zero variance edge case (e.g., all runs returned exactly the same score)
        if np.isclose(np.min(bootstrap_means), np.max(bootstrap_means)):
             ax.axvline(observed_mean, color=color, linestyle='-', linewidth=3, alpha=0.5, label=label)
        else:
            ax.hist(bootstrap_means, bins=50, density=True, alpha=0.4, color=color, edgecolor='white')
        
        ax.axvline(observed_mean, color=color, linestyle='--', linewidth=2, label=f'{label} (avg: {observed_mean:.1f})')
        ax.axvline(ci_lower, color=color, linestyle=':', linewidth=2)
        ax.axvline(ci_upper, color=color, linestyle=':', linewidth=2)

    if TR:
        ax.set_xlabel('Total Reward (Area Under Curve)', fontsize=16)
    else:
        ax.set_xlabel('Avg. Episode Return (AP)', fontsize=16)

    ax.set_ylabel('Density', fontsize=16)
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
    labels = [exp.get('label', f'Exp {i+1}') for i, exp in enumerate(experiments)]
    statistic = "TR" if TR else "AP"
    
    if len(data) < 2:
        print("Need at least 2 experiments for Welch test.")
        return

    # Filter out empty data
    valid_indices = [i for i, d in enumerate(data) if len(d) > 0]
    if len(valid_indices) < 2:
        print("Not enough valid data for Welch test.")
        return

    agent_1_stats = data[valid_indices[0]]
    agent_2_stats = data[valid_indices[1]]
    l1 = labels[valid_indices[0]]
    l2 = labels[valid_indices[1]]

    t_statistic, p_value = stats.ttest_ind(agent_1_stats, agent_2_stats, equal_var=False)

    print("=" * 50)
    print(f"Welch's t-test: {l1} vs {l2} ({statistic})")
    print("-" * 50)
    print(f"Mean {l1}: {np.mean(agent_1_stats):.4f}")
    print(f"Mean {l2}: {np.mean(agent_2_stats):.4f}")
    print(f"t-stat: {t_statistic:.4f}")
    print(f"p-val: {p_value:.6f}")
    
    if p_value < 0.05:
        print(f"Result: SIGNIFICANT difference (p < 0.05)")
    else:
        print(f"Result: NO significant difference")
    print("-" * 50)