"""
Statistical Analysis Script for MARL Experiment Results

This script loads experiment results from eval_returns.csv files and performs
statistical tests to compare different algorithm configurations.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ExperimentResults:
    """Container for a single experiment's results."""
    
    def __init__(self, path: str, name: str = None):
        self.path = path
        self.name = name or Path(path).name
        self.df = pd.read_csv(os.path.join(path, "eval_returns.csv"))
        
    @property
    def agent1_final_mean(self) -> float:
        """Get final evaluation mean for agent 1."""
        return self.df['agent1_mean'].iloc[-1]
    
    @property
    def agent2_final_mean(self) -> float:
        """Get final evaluation mean for agent 2."""
        return self.df['agent2_mean'].iloc[-1]
    
    @property
    def agent1_means(self) -> np.ndarray:
        """Get all evaluation means for agent 1."""
        return self.df['agent1_mean'].values
    
    @property
    def agent2_means(self) -> np.ndarray:
        """Get all evaluation means for agent 2."""
        return self.df['agent2_mean'].values
    
    def summary(self) -> Dict:
        """Generate summary statistics for this experiment."""
        return {
            'name': self.name,
            'n_evals': len(self.df),
            'agent1_final_mean': self.agent1_final_mean,
            'agent1_final_std': self.df['agent1_std'].iloc[-1],
            'agent2_final_mean': self.agent2_final_mean,
            'agent2_final_std': self.df['agent2_std'].iloc[-1],
            'agent1_max_mean': self.df['agent1_mean'].max(),
            'agent2_max_mean': self.df['agent2_mean'].max(),
            'agent1_avg_mean': self.df['agent1_mean'].mean(),
            'agent2_avg_mean': self.df['agent2_mean'].mean(),
        }


def discover_experiments(base_path: str) -> Dict[str, List[ExperimentResults]]:
    """
    Discover all experiments in the output directory.
    Returns a dict with category (IQL, MixedPlay, etc.) as keys.
    """
    experiments = {}
    base = Path(base_path)
    
    for category_dir in base.iterdir():
        if category_dir.is_dir() and category_dir.name != '__pycache__':
            if category_dir.name.endswith('.py'):
                continue
            experiments[category_dir.name] = []
            for exp_dir in category_dir.iterdir():
                if exp_dir.is_dir():
                    csv_path = exp_dir / "eval_returns.csv"
                    if csv_path.exists():
                        experiments[category_dir.name].append(
                            ExperimentResults(str(exp_dir))
                        )
    
    return experiments


def welch_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Perform Welch's t-test (unequal variances t-test).
    Returns t-statistic and p-value.
    """
    result = stats.ttest_ind(x, y, equal_var=False)
    return result.statistic, result.pvalue


def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Perform Mann-Whitney U test (non-parametric).
    Returns U-statistic and p-value.
    """
    result = stats.mannwhitneyu(x, y, alternative='two-sided')
    return result.statistic, result.pvalue


def paired_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Perform paired t-test for matched samples.
    Returns t-statistic and p-value.
    """
    result = stats.ttest_rel(x, y)
    return result.statistic, result.pvalue


def wilcoxon_test(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric paired test).
    Returns statistic and p-value.
    """
    try:
        result = stats.wilcoxon(x, y)
        return result.statistic, result.pvalue
    except ValueError:
        # Wilcoxon requires at least one non-zero difference
        return np.nan, np.nan


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.
    """
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + 
                          (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0


def compare_experiments(exp1: ExperimentResults, exp2: ExperimentResults, 
                        agent: int = 1) -> Dict:
    """
    Compare two experiments using multiple statistical tests.
    
    Args:
        exp1: First experiment
        exp2: Second experiment
        agent: Which agent to compare (1 or 2)
    
    Returns:
        Dictionary with test results
    """
    if agent == 1:
        x = exp1.agent1_means
        y = exp2.agent1_means
    else:
        x = exp1.agent2_means
        y = exp2.agent2_means
    
    # Ensure same length for paired tests
    min_len = min(len(x), len(y))
    x_paired = x[:min_len]
    y_paired = y[:min_len]
    
    t_stat, t_pval = welch_ttest(x, y)
    u_stat, u_pval = mann_whitney_u(x, y)
    paired_t_stat, paired_t_pval = paired_ttest(x_paired, y_paired)
    w_stat, w_pval = wilcoxon_test(x_paired, y_paired)
    effect_size = cohens_d(x, y)
    
    return {
        'exp1_name': exp1.name,
        'exp2_name': exp2.name,
        'agent': agent,
        'exp1_mean': np.mean(x),
        'exp2_mean': np.mean(y),
        'exp1_std': np.std(x),
        'exp2_std': np.std(y),
        'welch_t_stat': t_stat,
        'welch_t_pval': t_pval,
        'mann_whitney_u': u_stat,
        'mann_whitney_pval': u_pval,
        'paired_t_stat': paired_t_stat,
        'paired_t_pval': paired_t_pval,
        'wilcoxon_stat': w_stat,
        'wilcoxon_pval': w_pval,
        'cohens_d': effect_size,
        'significant_005': t_pval < 0.05,
        'significant_001': t_pval < 0.01,
    }


def compare_agents_within_experiment(exp: ExperimentResults) -> Dict:
    """
    Compare agent 1 vs agent 2 within the same experiment.
    """
    x = exp.agent1_means
    y = exp.agent2_means
    
    t_stat, t_pval = paired_ttest(x, y)
    w_stat, w_pval = wilcoxon_test(x, y)
    effect_size = cohens_d(x, y)
    
    return {
        'experiment': exp.name,
        'agent1_mean': np.mean(x),
        'agent2_mean': np.mean(y),
        'agent1_std': np.std(x),
        'agent2_std': np.std(y),
        'paired_t_stat': t_stat,
        'paired_t_pval': t_pval,
        'wilcoxon_stat': w_stat,
        'wilcoxon_pval': w_pval,
        'cohens_d': effect_size,
        'winner': 'agent1' if np.mean(x) > np.mean(y) else 'agent2',
        'significant': t_pval < 0.05,
    }


def print_comparison_results(results: Dict, verbose: bool = True):
    """Pretty print comparison results."""
    print(f"\n{'='*70}")
    print(f"Comparing: {results['exp1_name']} vs {results['exp2_name']}")
    print(f"Agent: {results['agent']}")
    print(f"{'='*70}")
    
    print(f"\nDescriptive Statistics:")
    print(f"  Experiment 1: mean = {results['exp1_mean']:.4f}, std = {results['exp1_std']:.4f}")
    print(f"  Experiment 2: mean = {results['exp2_mean']:.4f}, std = {results['exp2_std']:.4f}")
    
    print(f"\nStatistical Tests:")
    print(f"  Welch's t-test:    t = {results['welch_t_stat']:+.4f}, p = {results['welch_t_pval']:.4f}")
    print(f"  Mann-Whitney U:    U = {results['mann_whitney_u']:.4f}, p = {results['mann_whitney_pval']:.4f}")
    print(f"  Paired t-test:     t = {results['paired_t_stat']:+.4f}, p = {results['paired_t_pval']:.4f}")
    if not np.isnan(results['wilcoxon_stat']):
        print(f"  Wilcoxon:          W = {results['wilcoxon_stat']:.4f}, p = {results['wilcoxon_pval']:.4f}")
    
    print(f"\nEffect Size:")
    print(f"  Cohen's d = {results['cohens_d']:.4f}", end="")
    d = abs(results['cohens_d'])
    if d < 0.2:
        print(" (negligible)")
    elif d < 0.5:
        print(" (small)")
    elif d < 0.8:
        print(" (medium)")
    else:
        print(" (large)")
    
    print(f"\nSignificance:")
    print(f"  p < 0.05: {'Yes ✓' if results['significant_005'] else 'No'}")
    print(f"  p < 0.01: {'Yes ✓' if results['significant_001'] else 'No'}")


def print_agent_comparison(results: Dict):
    """Pretty print within-experiment agent comparison."""
    print(f"\n{'='*70}")
    print(f"Agent Comparison: {results['experiment']}")
    print(f"{'='*70}")
    
    print(f"\nDescriptive Statistics:")
    print(f"  Agent 1: mean = {results['agent1_mean']:.4f}, std = {results['agent1_std']:.4f}")
    print(f"  Agent 2: mean = {results['agent2_mean']:.4f}, std = {results['agent2_std']:.4f}")
    
    print(f"\nStatistical Tests:")
    print(f"  Paired t-test: t = {results['paired_t_stat']:+.4f}, p = {results['paired_t_pval']:.4f}")
    if not np.isnan(results['wilcoxon_stat']):
        print(f"  Wilcoxon:      W = {results['wilcoxon_stat']:.4f}, p = {results['wilcoxon_pval']:.4f}")
    
    print(f"\nEffect Size: Cohen's d = {results['cohens_d']:.4f}")
    print(f"Winner: {results['winner']} (significant: {'Yes' if results['significant'] else 'No'})")


def generate_summary_table(experiments: Dict[str, List[ExperimentResults]]) -> pd.DataFrame:
    """Generate a summary table of all experiments."""
    summaries = []
    for category, exp_list in experiments.items():
        for exp in exp_list:
            summary = exp.summary()
            summary['category'] = category
            summaries.append(summary)
    
    return pd.DataFrame(summaries)


# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    # Get the directory where this script is located
    SCRIPT_DIR = Path(__file__).parent
    
    print("=" * 70)
    print("MARL Statistical Analysis")
    print("=" * 70)
    
    # Discover all experiments
    experiments = discover_experiments(str(SCRIPT_DIR))
    
    print(f"\nDiscovered {sum(len(v) for v in experiments.values())} experiments:")
    for category, exp_list in experiments.items():
        print(f"\n  {category}:")
        for exp in exp_list:
            print(f"    - {exp.name}")
    
    # Generate summary table
    print("\n" + "=" * 70)
    print("Summary Table")
    print("=" * 70)
    summary_df = generate_summary_table(experiments)
    print(summary_df.to_string())
    
    # Example: Compare agents within MixedPlay experiments
    if 'MixedPlay' in experiments and experiments['MixedPlay']:
        print("\n" + "=" * 70)
        print("Agent Comparisons within MixedPlay Experiments")
        print("=" * 70)
        
        for exp in experiments['MixedPlay']:
            results = compare_agents_within_experiment(exp)
            print_agent_comparison(results)
    
    # Example: Compare different algorithms if available
    if 'MixedPlay' in experiments and len(experiments['MixedPlay']) >= 2:
        print("\n" + "=" * 70)
        print("Cross-Experiment Comparisons (Agent 1)")
        print("=" * 70)
        
        exp_list = experiments['MixedPlay']
        # Compare first two experiments as an example
        results = compare_experiments(exp_list[0], exp_list[1], agent=1)
        print_comparison_results(results)
    
    # Save summary to CSV
    output_path = SCRIPT_DIR / "analysis_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"\n\nSummary saved to: {output_path}")
