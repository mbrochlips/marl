import matplotlib.pyplot as plt
import numpy as np
import csv
import os


FIG_WIDTH=5
FIG_HEIGHT=2
FIG_ALPHA=0.2
FIG_WSPACE=0.3
FIG_HSPACE=0.2


def visualise_q_tables(q_tables, max_states=10, show_all_actions=True):
    """
    Visualize Q-tables for all agents.
    
    Handles various Q-table structures:
    - IQL: keys are str((state, action))
    - JAL/joint: keys may be str((state, own_action, other_actions)) or similar
    
    :param q_tables: List of Q-tables (defaultdicts), one per agent
    :param max_states: Maximum number of unique states to display per agent
    :param show_all_actions: If True, group by state and show all action Q-values together
    """
    from collections import defaultdict
    import ast
    
    for i, q_table in enumerate(q_tables):
        print(f"\n{'='*60}")
        print(f"Q-table for Agent {i + 1}")
        print(f"{'='*60}")
        
        if not q_table:
            print("  (empty)")
            continue
        
        # Group entries by state
        # Key format: str((state, action)) or str((state, action, other_action))
        state_action_values = defaultdict(dict)
        
        for key_str, q_value in q_table.items():
            try:
                # Parse the string key back to tuple
                key_tuple = ast.literal_eval(key_str)
                
                if len(key_tuple) == 2:
                    # IQL format: (state, action)
                    state, action = key_tuple
                    state_key = str(state)
                    action_key = action
                elif len(key_tuple) == 3:
                    # Joint action format: (state, own_action, other_action)
                    state, own_action, other_action = key_tuple
                    state_key = f"{state} | other={other_action}"
                    action_key = own_action
                else:
                    # Unknown format - use as-is
                    state_key = str(key_tuple[:-1])
                    action_key = key_tuple[-1]
                
                state_action_values[state_key][action_key] = q_value
            except:
                # Fallback for unparseable keys
                state_action_values["unparsed"][key_str] = q_value
        
        # Sort states by max Q-value (most valuable states first)
        sorted_states = sorted(
            state_action_values.items(),
            key=lambda x: max(x[1].values()) if x[1] else 0,
            reverse=True
        )
        
        # Display
        displayed_states = 0
        for state_key, action_values in sorted_states:
            if displayed_states >= max_states:
                remaining = len(sorted_states) - displayed_states
                print(f"\n  ... and {remaining} more states")
                break
            
            print(f"\n  State: {state_key}")
            
            # Sort actions by Q-value
            sorted_actions = sorted(action_values.items(), key=lambda x: x[1], reverse=True)
            best_action = sorted_actions[0][0] if sorted_actions else None
            
            for action, q_val in sorted_actions:
                marker = " *" if action == best_action else ""
                print(f"    Action {action}: {q_val:8.4f}{marker}")
            
            displayed_states += 1
        
        # Summary statistics
        if q_table:
            values = list(q_table.values())
            print(f"\n  {'─'*40}")
            print(f"  Summary: {len(q_table)} entries across {len(state_action_values)} states")
            print(f"  Q-range: [{min(values):.4f}, {max(values):.4f}]")
            print(f"  Mean Q: {np.mean(values):.4f}, Std: {np.std(values):.4f}")


def visualise_evaluation_returns(means, stds, config, dirpath:str):
    """
    Plot evaluation returns

    :param means (List[List[float]]): mean evaluation returns for each agent
    :param stds (List[List[float]]): standard deviation of evaluation returns for each agent
    """
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
                writer.writerow(header)

                for idx in range(n_evals):
                    row = [idx]
                    for a in range(n_agents):
                        row.append(float(means[idx][a]))
                        row.append(float(stds[idx][a]))
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
