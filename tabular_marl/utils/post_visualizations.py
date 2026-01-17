import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
import os

# Figure settings
FIG_WIDTH = 5
FIG_HEIGHT = 2
FIG_ALPHA = 0.2

def load_eval_returns_from_csv(csv_path):
    """
    Load evaluation returns from a CSV file.
    
    :param csv_path: Path to the eval_returns_full.csv file
    :return: numpy array of shape (n_reps, total_eval_eps, n_agents)
    """
    df = pd.read_csv(csv_path)
    
    all_returns = []
    for _, row in df.iterrows():
        agent1_returns = ast.literal_eval(row['agent_1_returns'])
        agent2_returns = ast.literal_eval(row['agent_2_returns'])
        rep_returns = np.column_stack([agent1_returns, agent2_returns])
        all_returns.append(rep_returns)
    
    return np.array(all_returns)  # (n_reps, total_eval_eps, n_agents)


def get_color_from_agent_name(run_path):
    """
    Extract the first agent name from the run_path and return the corresponding color.
    
    Color mapping:
    - IQL: red
    - JAL-AM (JalAM): blue  
    - IQL-AE (IQLAE): orange
    - JAL-AE (JalAE): purple
    - QBM: green
    """
    # Extract folder name from path
    filename = os.path.basename(run_path.rstrip('/'))
    
    # Get the first agent name (before _vs_)
    first_agent = filename.split('_vs_')[0] if '_vs_' in filename else filename.split('_')[0]
    second_agent = filename.split('_vs_')[1].split("_")[0]

    first_agent_lower = first_agent.lower()
    second_agent_lower = second_agent.lower()
    
    # Color mapping based on agent name
    color_map = {
        'iql': '#D32F2F',      # Red
        'jalam': '#1976D2',    # Blue
        'jalunce': '#1976D2',    # Blue (JAL with uncertainty estimation)
        'iqlae': "#7B1FA2",    # Orange
        'jalae': '#7B1FA2',    # Purple
        'qbm': '#388E3C',      # Green
        'prandom': '#757575',  # Grey for random baseline
    }
    
    # Try to match the agent name
    for key, color in color_map.items():
        if key == first_agent_lower:
            if key == 'prandom':
                continue
            return color

    for key, color in color_map.items():
        if key in second_agent_lower:
            if key == 'prandom':
                continue
            return color
    
    # Default fallback color
    return '#607D8B'  # Blue Grey


def visualise_multiple_learning_curves(experiment_data, title=None, agent_idx=0):
    """
    Visualize multiple experiment learning curves on the same plot.
    Uses eval_returns_full.csv from each run_path.
    
    :param experiment_data: List of dicts, each containing:
        - 'run_path': path to the experiment run folder
        - 'label': label for the legend
        - 'color': (optional) color for the line - auto-detected from agent name if not provided
        - 'total_eps': total training episodes
        - 'ep_length': episode length
        - 'n_checkpoints': number of evaluation checkpoints (default 10)
    :param title: Optional title for the plot
    :param agent_idx: Which agent to plot (0 for agent 1, 1 for agent 2). 
                      If None, plots combined mean of all agents.
    :return: matplotlib figure
    """
    # Fallback colors if agent name not recognized
    fallback_colors = [
        "#E64A19",  # Deep Orange
        "#1976D2",  # Blue
        "#388E3C",  # Green  
        "#7B1FA2",  # Purple
        "#F57C00",  # Orange
        "#0097A7",  # Cyan
        "#C2185B",  # Pink
        "#FBC02D",  # Yellow
    ]
    
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.5, FIG_HEIGHT * 2.5))
    
    for i, exp in enumerate(experiment_data):
        # Load data from eval_returns_full.csv
        run_path = exp['run_path']
        csv_path = os.path.join(run_path, 'eval_returns_full.csv')
        all_returns = load_eval_returns_from_csv(csv_path)
        n_reps, total_eval_eps, n_agents = all_returns.shape
        
        # Get config values
        total_eps = exp.get('total_eps', 300)
        ep_length = exp.get('ep_length', 50)
        n_checkpoints = exp.get('n_checkpoints', 10)
        
        # Calculate checkpoint episodes (evenly spaced: 10%, 20%, ..., 100%)
        checkpoint_pcts = list(range(10, 101, 10))[:n_checkpoints]
        checkpoint_eps = [int(pct / 100 * total_eps) for pct in checkpoint_pcts]
        
        # Calculate eval episodes per checkpoint
        eval_eps_per_checkpoint = total_eval_eps // n_checkpoints
        
        # Select which agent to plot
        if agent_idx is not None:
            agent_returns = all_returns[:, :, agent_idx]
        else:
            # Combined mean across all agents
            agent_returns = np.mean(all_returns, axis=2)
        
        # Reshape to group by checkpoint: (n_reps, n_checkpoints, eval_eps_per_checkpoint)
        agent_returns_reshaped = agent_returns.reshape(n_reps, n_checkpoints, eval_eps_per_checkpoint)
        
        # Mean within each checkpoint: (n_reps, n_checkpoints)
        checkpoint_returns = np.mean(agent_returns_reshaped, axis=2)
        
        # Mean and std across repetitions: (n_checkpoints,)
        checkpoint_means = np.mean(checkpoint_returns, axis=0)
        checkpoint_stds = np.std(checkpoint_returns, axis=0)
        
        # Get color - auto-detect from agent name if not provided
        if 'color' in exp:
            color = exp['color']
        else:
            color = get_color_from_agent_name(run_path)
        label = exp.get('label', f'Experiment {i+1}')
        
        # Plot mean line
        ax.plot(checkpoint_eps[:len(checkpoint_means)], checkpoint_means,
                color=color, linewidth=2, marker='o', markersize=5, label=label)
        
        # Plot std band
        ax.fill_between(checkpoint_eps[:len(checkpoint_means)],
                        checkpoint_means - checkpoint_stds,
                        checkpoint_means + checkpoint_stds,
                        alpha=FIG_ALPHA, color=color)
    
    ax.set_xlabel('Episode', fontsize=16)
    ax.set_ylabel('Gns. Afkast', fontsize=16)
    if title:
        ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=12)
    ax.tick_params(labelsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def visualise_end_returns_comparison_old(experiment_data, title=None):
    """
    Visualize end-returns (final checkpoint) from multiple experiments in adjacent plots.
    Each experiment gets its own subplot showing the distribution of final returns across repetitions.
    Uses eval_returns_last10.csv from each run_path.
    
    :param experiment_data: List of dicts, each containing:
        - 'run_path': path to the experiment run folder
        - 'label': label for the plot title
        - 'color': (optional) color for the bars - auto-detected from agent name if not provided
        - 'n_checkpoints': number of evaluation checkpoints (default 10)
    :param title: Optional overall title for the figure
    :return: matplotlib figure
    """
    n_experiments = len(experiment_data)
    
    fig, axes = plt.subplots(1, n_experiments, figsize=(FIG_WIDTH * n_experiments, FIG_HEIGHT * 2.5))
    if n_experiments == 1:
        axes = [axes]
    
    for idx, exp in enumerate(experiment_data):

        run_path = exp['run_path']
        csv_path = os.path.join(run_path, "eval_returns_last10.csv")
        
        all_returns = load_eval_returns_from_csv(csv_path)
        n_reps, total_eval_eps, n_agents = all_returns.shape

        expected_total_eval_eps =  exp.get("total_eval_eps", 100)

        # If to many eval_eps
        if total_eval_eps != expected_total_eval_eps:
            stepsize = total_eval_eps // expected_total_eval_eps
            all_returns = all_returns[:, ::stepsize, :]
 
        # Collapse the agent dimension first
        if run_path.startswith("pRand"):
            # Assuming Agent 1 (index 1) is the learner against random
            data_agents = all_returns[:, :, 1] 
        else:
            # Collaborative/Self-play: Mean across agents
            data_agents = np.mean(all_returns, axis=2)

        # --- AVERAGE PERFORMANCE (Robust Final Estimate) ---
        # The file contains a block of episodes representing the final state.
        # We average over ALL episodes in this file for each repetition.
        # Shape: (n_reps,)
        combined_rep_means = np.mean(data_agents, axis=1)

        # Overall statistics
        overall_mean = np.mean(combined_rep_means)
        overall_std = np.std(combined_rep_means)
        
        # Get color
        if 'color' in exp:
            color = exp['color']
        else:
            color = get_color_from_agent_name(run_path)
        
        label = exp.get('label', f'Experiment {idx+1}')
        
        # Plot individual repetition means as scatter
        ax.scatter(range(len(combined_rep_means)), combined_rep_means, 
                   alpha=0.6, color=color, s=40, zorder=3)
        
        # Plot overall mean as horizontal line
        ax.axhline(overall_mean, color=color, linestyle='-', linewidth=2)
        print(f'Gns.: {overall_mean:.2f}, Std: ±{overall_std:.2f}')
        # Plot std band
        ax.axhspan(overall_mean - overall_std, overall_mean + overall_std, 
                   alpha=FIG_ALPHA, color=color)
        
        ax.set_xlabel('Repetition', fontsize=14)
        ax.set_ylabel('Gns. Slutafkast', fontsize=14)
        ax.set_title(label, fontsize=16)
        ax.legend(loc='best', fontsize=10)
        ax.tick_params(labelsize=12)
        ax.set_xlim(-0.5, len(combined_rep_means) - 0.5)
        ax.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=18, y=1.02)
    
    # Collect all values to determine global y-limits
    all_values = []
    for exp in experiment_data:
        # ...existing code to compute combined_rep_means...
        all_values.extend(combined_rep_means)
    diff = max(all_values) - min(all_values)
    ymin = min(all_values) - 0.1 * diff
    ymax = max(all_values) + 0.1 * diff

    # After plotting, set the same y-limits for all axes
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    
    return fig


import matplotlib.pyplot as plt
import numpy as np
import os

def visualise_end_returns_comparison(experiment_data, title=None):
    n_experiments = len(experiment_data)
    
    fig, axes = plt.subplots(1, n_experiments, figsize=(FIG_WIDTH * n_experiments, FIG_HEIGHT * 2.5))
    
    # Ensure axes is always iterable
    if n_experiments == 1:
        axes = [axes]
    
    # ### FIX: List to store all data points for global y-limits later
    all_global_values = []

    for idx, exp in enumerate(experiment_data):
        # ### FIX: Define 'ax' for the current iteration
        ax = axes[idx]

        run_path = exp['run_path']
        csv_path = os.path.join(run_path, "eval_returns_last10.csv")
        
        # Check if file exists to prevent crash
        if not os.path.exists(csv_path):
            print(f"Warning: File not found {csv_path}")
            continue

        all_returns = load_eval_returns_from_csv(csv_path)
        n_reps, total_eval_eps, n_agents = all_returns.shape

        expected_total_eval_eps = exp.get("total_eval_eps", 100)

        # If too many eval_eps, subsample
        if total_eval_eps != expected_total_eval_eps:
            stepsize = total_eval_eps // expected_total_eval_eps
            all_returns = all_returns[:, ::stepsize, :]
 
        # Collapse the agent dimension
        if run_path.startswith("pRand"):
            data_agents = all_returns[:, :, 1] 
        else:
            data_agents = np.mean(all_returns, axis=2)

        # Average over all episodes in the file per repetition
        combined_rep_means = np.mean(data_agents, axis=1)

        # ### FIX: Collect values for global scaling later
        all_global_values.extend(combined_rep_means)

        # Overall statistics
        overall_mean = np.mean(combined_rep_means)
        overall_std = np.std(combined_rep_means)
        print(f"Global Mean: {overall_mean}")
        print(f"Global std: {overall_std}")

        # Get color
        if 'color' in exp:
            color = exp['color']
        else:
            color = get_color_from_agent_name(run_path) 
        
        label = exp.get('label', f'Experiment {idx+1}')
        
        # Plotting
        ax.scatter(range(len(combined_rep_means)), combined_rep_means, 
                   alpha=0.6, color=color, s=40, zorder=3)
        
        ax.axhline(overall_mean, color=color, linestyle='-', linewidth=2)
        
        # Plot std band
        ax.axhspan(overall_mean - overall_std, overall_mean + overall_std, 
                   alpha=FIG_ALPHA, color=color)
        
        ax.set_xlabel('Repetition', fontsize=14)
        ax.set_ylabel('Gns. Slutafkast', fontsize=14)
        ax.set_title(label, fontsize=16)
        
        # Only add legend if needed, usually 'label' in scatter handles this 
        # but here you are creating custom lines. 
        # If you want a legend, ensure elements have 'label' props.
        # ax.legend(loc='best', fontsize=10) 
        
        ax.tick_params(labelsize=12)
        ax.set_xlim(-0.5, len(combined_rep_means) - 0.5)
        ax.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=18, y=1.02)
    
    # ### FIX: Calculate limits based on the data collected during the first loop
    if all_global_values:
        diff = max(all_global_values) - min(all_global_values)
        if diff == 0: diff = 1.0 # Prevent crash if all values are identical
        ymin = min(all_global_values) - 0.1 * diff
        ymax = max(all_global_values) + 0.1 * diff

        # Apply limits to all axes
        for ax in axes:
            ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    return fig