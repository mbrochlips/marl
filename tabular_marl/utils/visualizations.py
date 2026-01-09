import matplotlib.pyplot as plt
import numpy as np
import csv
import os


FIG_WIDTH=5
FIG_HEIGHT=2
FIG_ALPHA=0.2
FIG_WSPACE=0.3
FIG_HSPACE=0.2


def visualise_q_tables(q_tables, save_path=None, max_states=None):
    """
    Visualize Q-tables for all agents and save to file.
    
    Handles various Q-table structures:
    - IQL: keys are str((state, action))
    - JAL/joint: keys may be str((state, own_action, other_actions)) or similar
    
    :param q_tables: List of Q-tables (defaultdicts), one per agent
    :param save_path: Directory path to save Q-table files. If None, prints to terminal.
    :param max_states: Maximum number of unique states to display per agent. None = show all.
    """
    from collections import defaultdict
    
    for i, q_table in enumerate(q_tables):
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"Q-table for Agent {i + 1}")
        lines.append(f"{'='*70}")
        
        if not q_table:
            lines.append("  (empty)")
            _output_lines(lines, save_path, f"q_table_agent_{i+1}.txt")
            continue
        
        # Group entries by state
        # Key format: str((state, action)) or str((state, action, other_action))
        # State can be a list or numpy array
        state_action_values = defaultdict(dict)
        unique_base_states = set()  # Track actual unique states (without action info)
        
        for key_str, q_value in q_table.items():
            state_key, action_key, base_state = _parse_q_key(key_str)
            state_action_values[state_key][action_key] = q_value
            if base_state:
                unique_base_states.add(base_state)
        
        # Sort states by max Q-value (most valuable states first)
        sorted_states = sorted(
            state_action_values.items(),
            key=lambda x: max(x[1].values()) if x[1] else 0,
            reverse=True
        )
        
        # Summary statistics first
        values = list(q_table.values())
        lines.append(f"\nSUMMARY")
        lines.append(f"{'─'*70}")
        lines.append(f"  Total entries: {len(q_table)}")
        lines.append(f"  Unique states visited: {len(unique_base_states)}")
        lines.append(f"  Q-value range: [{min(values):.4f}, {max(values):.4f}]")
        lines.append(f"  Mean Q-value:  {np.mean(values):.4f}")
        lines.append(f"  Std Q-value:   {np.std(values):.4f}")
        lines.append("")
        
        # Display states
        lines.append(f"STATES (sorted by max Q-value)")
        lines.append(f"{'─'*70}")
        
        displayed_states = 0
        for state_key, action_values in sorted_states:
            if max_states is not None and displayed_states >= max_states:
                remaining = len(sorted_states) - displayed_states
                lines.append(f"\n  ... and {remaining} more states (use max_states=None to show all)")
                break
            
            lines.append(f"\n  State: {state_key}")
            
            # Sort actions by Q-value
            sorted_actions = sorted(action_values.items(), key=lambda x: x[1], reverse=True)
            best_action = sorted_actions[0][0] if sorted_actions else None
            
            for action, q_val in sorted_actions:
                marker = " <-- best" if action == best_action else ""
                lines.append(f"    Action {action}: {q_val:10.4f}{marker}")
            
            displayed_states += 1
        
        _output_lines(lines, save_path, f"q_table_agent_{i+1}.txt")


def _parse_q_key(key_str):
    """
    Parse Q-table key string into (state_key, action_key, base_state).
    
    Handles formats:
    - IQL with lists: "([1, 0, 1, 0], 2)"
    - IQL with numpy: "(array([1., 0., 1., 0.], dtype=float32), 2)"
    - JAL with lists: "([1, 0, 1, 0], 2, 3)"
    - JAL with numpy: "(array([...]), 2, 3)"
    
    Returns (state_key, action_key, base_state_str)
    """
    import re
    import ast
    
    try:
        # Try standard ast parsing first (works for list-based states)
        key_tuple = ast.literal_eval(key_str)
        if len(key_tuple) == 2:
            state, action = key_tuple
            state_str = str(state)
            return state_str, action, state_str
        elif len(key_tuple) == 3:
            state, own_action, opp_action = key_tuple
            state_str = str(state)
            return f"{state_str} | opp_action={opp_action}", own_action, state_str
        else:
            return str(key_tuple[:-1]), key_tuple[-1], str(key_tuple[0])
    except:
        pass
    
    # Handle numpy array format: "(array([...], dtype=...), action)" or "(array([...]), action, opp_action)"
    try:
        # Extract the array content
        array_match = re.search(r'array\(\[(.*?)\]', key_str, re.DOTALL)
        if array_match:
            array_content = array_match.group(1)
            # Clean up the array values (remove newlines, extra spaces)
            array_values = re.sub(r'\s+', ' ', array_content).strip()
            state_str = f"[{array_values}]"
            
            # Find actions at the end - look for numbers after the array closes
            # Format: "...), action)" or "...), action, opp_action)"
            after_array = key_str[key_str.rfind(']'):]
            numbers = re.findall(r'(\d+)', after_array)
            
            if len(numbers) >= 2:
                # JAL format: (state, own_action, opp_action)
                own_action = int(numbers[-2])
                opp_action = int(numbers[-1])
                return f"{state_str} | opp_action={opp_action}", own_action, state_str
            elif len(numbers) == 1:
                # IQL format: (state, action)
                action = int(numbers[0])
                return state_str, action, state_str
    except:
        pass
    
    # Final fallback
    return "unparsed", key_str, None


def _output_lines(lines, save_path, filename):
    """Helper to either print lines or save to file."""
    content = "\n".join(lines)
    
    if save_path is None:
        print(content)
    else:
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, filename)
        with open(filepath, "w") as f:
            f.write(content)
        print(f"Saved: {filepath}")


def visualise_repetition_returns(all_eval_returns, config):
    """
    Visualize evaluation returns across multiple repetitions.
    Shows mean ± std across repetitions for each agent.

    :param all_eval_returns: List[List[np.array]] - shape (repetitions, eval_episodes_per_rep, num_agents)
    :param config: Configuration dictionary with algorithm names
    """
    # Convert to numpy array: (repetitions, eval_episodes, num_agents)
    all_returns = np.array(all_eval_returns)
    n_reps, n_eval_eps, n_agents = all_returns.shape
    
    # Calculate mean per repetition: (repetitions, num_agents)
    rep_means = np.mean(all_returns, axis=1)
    
    # Mean and std across repetitions
    overall_mean = np.mean(rep_means, axis=0)  # (num_agents,)
    overall_std = np.std(rep_means, axis=0)    # (num_agents,)
    
    # Create figure
    fig, axes = plt.subplots(1, n_agents, figsize=(FIG_WIDTH * n_agents, FIG_HEIGHT * 2))
    if n_agents == 1:
        axes = [axes]
    
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]  # Blue, Red, Green, Orange
    algorithm_names = [config.get('algorithm_1', 'Agent 1'), config.get('algorithm_2', 'Agent 2')]
    
    for i in range(n_agents):
        ax = axes[i]
        agent_rep_means = rep_means[:, i]  # (repetitions,)
        
        # Plot individual repetition means as scatter
        ax.scatter(range(len(agent_rep_means)), agent_rep_means, 
                   alpha=0.5, color=colors[i % len(colors)], s=30)
        
        # Plot overall mean as horizontal line
        ax.axhline(overall_mean[i], color=colors[i % len(colors)], 
                   linestyle='-', linewidth=2, label=f'Mean: {overall_mean[i]:.2f}')
        
        # Plot std band
        ax.axhspan(overall_mean[i] - overall_std[i], overall_mean[i] + overall_std[i], 
                   alpha=FIG_ALPHA, color=colors[i % len(colors)], 
                   label=f'Std: ±{overall_std[i]:.2f}')
        
        ax.set_xlabel('Repetition', fontsize=14)
        ax.set_ylabel('Mean Evaluation Return', fontsize=14)
        ax.set_title(f'{algorithm_names[i]} (Agent {i+1})', fontsize=16)
        ax.legend(loc='upper left', fontsize=12)
        ax.tick_params(labelsize=14)
        ax.set_xlim(-0.5, len(agent_rep_means) - 0.5)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def visualise_learning_curve(all_eval_returns, config):
    """
    Visualize learning curve: mean return across repetitions at each evaluation checkpoint.
    Shows how performance evolves during training.

    :param all_eval_returns: List[List[np.array]] - shape (repetitions, total_eval_eps, num_agents)
                             where total_eval_eps = num_checkpoints * eval_eps_per_checkpoint
    :param config: Configuration dictionary with algorithm names and eval settings
    """
    # Convert to numpy array: (repetitions, total_eval_eps, num_agents)
    all_returns = np.array(all_eval_returns)
    n_reps, total_eval_eps, n_agents = all_returns.shape
    
    # Determine number of checkpoints and checkpoint percentages based on eval_spread
    eval_spread = config.get("eval_spread")

    total_eps = config.get("total_eps")
    ep_length = config.get("ep_length")
    max_steps = total_eps * ep_length
    
    if eval_spread == "full":
        # Evenly spaced: 10%, 20%, ..., 100%
        n_checkpoints = 10
        checkpoint_pcts = list(range(10, 101, 10))
    else:
        # Last 10%: 91%, 92%, ..., 100%
        eval_start = int(0.9 * total_eps)
        n_checkpoints = total_eps - eval_start
        checkpoint_pcts = list(range(91, 101))[:n_checkpoints]
    
    # Convert checkpoint percentages to total steps
    checkpoint_steps = [int(pct / 100 * max_steps) for pct in checkpoint_pcts]
    
    # Calculate eval episodes per checkpoint
    eval_eps_per_checkpoint = total_eval_eps // n_checkpoints
    
    # Reshape to group by checkpoint: (repetitions, n_checkpoints, eval_eps_per_checkpoint, num_agents)
    # Then average within each checkpoint
    all_returns_reshaped = all_returns.reshape(n_reps, n_checkpoints, eval_eps_per_checkpoint, n_agents)
    
    # Mean within each checkpoint: (repetitions, n_checkpoints, num_agents)
    checkpoint_returns = np.mean(all_returns_reshaped, axis=2)
    
    # Mean and std across repetitions for each checkpoint: (n_checkpoints, num_agents)
    checkpoint_means = np.mean(checkpoint_returns, axis=0)
    checkpoint_stds = np.std(checkpoint_returns, axis=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.5, FIG_HEIGHT * 2.5))
    
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]  # Blue, Red, Green, Orange
    algorithm_names = [config.get('algorithm_1', 'Agent 1'), config.get('algorithm_2', 'Agent 2')]
    
    for i in range(n_agents):
        color = colors[i % len(colors)]
        means = checkpoint_means[:, i]
        stds = checkpoint_stds[:, i]
        
        # Plot mean line
        ax.plot(checkpoint_steps[:len(means)], means, 
                color=color, linewidth=2, marker='o', markersize=5,
                label=f'{algorithm_names[i]} (Agent {i+1})')
        
        # Plot std band
        ax.fill_between(checkpoint_steps[:len(means)], means - stds, means + stds,
                        alpha=FIG_ALPHA, color=color)
    
    ax.set_xlabel('Training Steps', fontsize=16)
    ax.set_ylabel('Mean Evaluation Return', fontsize=16)
    ax.legend(loc='best', fontsize=16)
    ax.tick_params(labelsize=16)
    step_margin = max_steps * 0.02  # 2% margin
    ax.set_xlim(checkpoint_steps[0] - step_margin, checkpoint_steps[-1] + step_margin)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def visualise_evaluation_returns(returns, config, dirpath:str):
    """
    Plot evaluation returns

    :param returns (List[List[List]]): returns for each agent [ [ [returni_eval1_1, returni_eval1_2,,,] , 
                                                                   [returnj_eval1_1, returnj_eval1_2,,,]] ,
                                                                [ [returni_eval1_1, returni_eval1_2,,,] , 
                                                                [returnj_eval1_1, returnj_eval1_2,,,]]
    """ 
    means = [np.mean(episodic_returns, axis=0) for episodic_returns in returns]
    stds = [np.std(episodic_returns, axis=0) for episodic_returns in returns]

    n_agents = len(means[0])
    n_evals = len(means)

    fig, ax = plt.subplots(nrows=1, ncols=n_agents, figsize=(FIG_WIDTH, FIG_HEIGHT * n_agents))

    colors = ["b", "r"]
    for i, color in enumerate(colors):
        ax[i].plot(range(n_evals), [mean[i] for mean in means], label=f"Agent {i+1}", color=color)
        ax[i].fill_between(range(n_evals), [mean[i] - std[i] for mean, std in zip(means, stds)],
                           [mean[i] + std[i] for mean, std in zip(means, stds)], alpha=FIG_ALPHA, color=color)
        ax[i].set_xlabel("Evaluations")
        ax[i].set_ylabel("Evaluation return")
    fig.legend()
    fig.subplots_adjust(hspace=FIG_HSPACE)
    if config["save"] == True:
        plt.savefig(f"{config['dir']}/eval_image.png", dpi=300, bbox_inches="tight")
        # save the data as a csv file:
        try:
            out_dir = config['dir']
            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, "eval_returns.csv")
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                # header: eval_index, agent1_mean, agent1_std, agent2_mean, agent2_std, ...
                header = ["eval_index"]
                for a in range(n_agents):
                    header.append(f"agent{a+1}_mean")
                    header.append(f"agent{a+1}_std")
                    header.append(f"agent{a+1}_returns")
                writer.writerow(header)

                for idx in range(n_evals):
                    row = [idx]
                    for a in range(n_agents):
                        row.append(float(means[idx][a]))
                        row.append(float(stds[idx][a]))
                        row.append(list(r[a] for r in returns[idx]))
                    writer.writerow(row)
        except Exception as e:
            # don't raise from plotting helper; just print a warning
            print(f"Warning: could not save eval returns csv: {e}")
    plt.show()

def visualise_q_convergence(eval_q_tables, env, savefig=None):
    """
    Plot q_table convergence
    :param eval_q_tables (List[List[Dict[Act, float]]]): q_tables of both agents for each evaluation
    :param env (gym.Env): gym matrix environment with `payoff` attribute
    :param savefig (str): path to save figure
    """
    assert hasattr(env, "payoff")
    payoff = np.array(env.payoff)
    n_actions = 2
    n_agents = 2
    assert payoff.shape == (n_actions, n_actions, n_agents), "Payoff matrix must be 2x2x2 for 2x2 PD game"
    # (n_evals, n_agents, n_actions)
    q_tables = np.array(
            [[[q_table[str((0, act))] for act in range(n_actions)] for q_table in q_tables] for q_tables in eval_q_tables]
    )

    fig, ax = plt.subplots(nrows=n_agents, ncols=n_actions, figsize=(n_actions * FIG_WIDTH, FIG_HEIGHT * n_agents))

    for i in range(n_agents):
        max_payoff = payoff[:, :, i].max()
        min_payoff = payoff[:, :, i].min()
    
        for act in range(n_actions):
            # plot max Q-values
            if i == 0:
                max_r = payoff[act, :, i].max()
                max_label = rf"$max_b Q(a, b)$"
                q_label = rf"$Q(a_{act}, \cdot)$"
            else:
                max_r = payoff[:, act, i].max()
                max_label = rf"$max_a Q(a, b_{act})$"
                q_label = rf"$Q(\cdot, b_{act})$"
            ax[i, act].axhline(max_r, ls='--', color='r', alpha=0.5, label=max_label)

            # plot respective Q-values
            q_values = q_tables[:, i, act]
            ax[i, act].plot(q_values, label=q_label)

            # axes labels and limits
            ax[i, act].set_ylim([min_payoff - 0.05, max_payoff + 0.05])
            ax[i, act].set_xlabel(f"Evaluations")
            if i == 0:
                ax[i, act].set_ylabel(fr"$Q(a_{act})$")
            else:
                ax[i, act].set_ylabel(fr"$Q(b_{act})$")

            ax[i, act].legend(loc="upper center")

    fig.subplots_adjust(wspace=FIG_WSPACE)

    if savefig is not None:
        plt.savefig(f"{savefig}.pdf", format="pdf")

    plt.show()
