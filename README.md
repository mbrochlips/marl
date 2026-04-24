# Bachelor Project 2026
This bachelor thesis investigates the challenges of stability and convergence in Multi-Agent Reinforcement Learning (MARL) within cooperative environments.
The study evaluates three algorithmic extensions: Joint Action Learning with Agent Modelling (JAL-AM), variance-based Active Exploration (AE), and Q-learning with Behavior Managing (QBM). Where AE and QBM are novel contributions.

GO to the folder "tabular_marl". This folder contains implementations of tabular multi-agent reinforcement learning algorithms for various environments.

## Overview

This codebase supports three main training paradigms:
- **Self-Play**: Agents of the same algorithm learn together
- **Mixed-Play**: Two different algorithms learn together
- **Multiple Runs**: Run the same configuration multiple times for robust evaluation

## Main Scripts

### `run.py` - Self-Play Training
Use this script for **Self-Play** purposes, where agents of the same algorithm learn together.

**Usage:**
```bash
python run.py
```

**Configuration:**
- Edit the `CONFIG` dictionary in `run.py` to customize:
  - Algorithm (e.g., "IQL")
  - Environment (e.g., "cf" for custom foraging, "f" for foraging, "m" for matrix game)
  - Training parameters (episodes, episode length, learning rate, etc.)

**Outputs:**
- Saved to `output/{ALGORITHM}/{runname}/`
- Includes evaluation returns CSV, visualizations, and videos (if enabled)

---

### `run_mixed.py` - Mixed-Play Training
Use this script for **Mixed-Play**, where two different algorithms learn together.

**Usage:**
```bash
python run_mixed.py
```

**Configuration:**
- Edit the `CONFIG` dictionary in `run_mixed.py` to customize:
  - `algorithm_1` and `algorithm_2`: Choose from available algorithms (IQL, IQLAE, JalAM, JalAE, Random, pRandom)
  - `algorithm_1_kwargs` and `algorithm_2_kwargs`: Additional parameters for each algorithm
  - Environment and training parameters

**Available Algorithms:**
- `IQL`: Independent Q-Learning
- `IQLAE`: IQL with uncertainty
- `JalAM`: Joint Action Learning with opponent modeling
- `JalAE`: JAL with uncertainty
- `QBM`: IQL with behaviour managing
- `Random`: Random agent
- `pRandom`: Probabilistic random agent

**Outputs:**
- Saved to `output/MixedPlay/{runname}/`
- Includes evaluation returns CSV, visualizations, and videos (if enabled)

---

### `run_multiple.py` - Multiple Repetitions
Use this script to run the same configuration of algorithms **multiple times** for robust evaluation.

**Usage:**
```bash
python run_multiple.py
```

**Configuration:**
- Edit the `CONFIG` dictionary in `run_multiple.py` to customize:
  - `repetitions`: Number of independent runs (e.g., 2, 30)
  - `algorithm_1` and `algorithm_2`: Algorithms to compare
  - `eval_spread`: Evaluation strategy ("last10", "full", or "both")
  - Environment and training parameters

**Features:**
- Runs multiple independent training repetitions
- Aggregates results across repetitions
- Provides statistical summaries (mean ± std)
- Generates learning curves and repetition comparison plots

**Outputs:**
- Saved to `output/Multiple/{runname}/` (or `output/Final/{runname}/` if 30 repetitions)
- Includes:
  - `eval_returns.csv` (or `eval_returns_full.csv` and `eval_returns_last10.csv` if `eval_spread="both"`)
  - `{runname}_repetition_returns.png`: Comparison across repetitions
  - `{runname}_learning_curve.png`: Learning curve visualization
  - Videos (if enabled)

---

## Folder Structure

```
tabular_marl/
├── agent/              # Agent implementations
│   ├── iql.py          # Independent Q-Learning
│   ├── iql_unc.py      # IQL with uncertainty
│   ├── jal.py          # Joint Action Learning
│   ├── jal_unc.py      # JAL with uncertainty
│   ├── random_agent.py # Random agent
│   ├── p_random.py    # Probabilistic random agent
│   ├── iql_behave_managing.py  # Q-learning with behavior management
│   └── mixed_play_wrapper.py   # Wrapper for mixed-play scenarios
│
├── envs/               # Environment implementations
│   ├── custom_foraging_env.py      # Custom foraging environment
│   ├── custom_foraging_oneFood.py  # Custom foraging with one food
│   ├── matrix_game.py               # Matrix game environment
│   ├── move_game.py                # Move chair game
│   ├── move_chair_simple.py        # Simple move chair game
│   └── move_game_coor.py           # Move chair coordination game
│
├── utils/              # Utility functions
│   ├── visualizations.py    # Plotting and visualization functions
│   ├── video.py             # Video recording utilities
│   ├── eval.py              # Evaluation functions
│   ├── post_stats.py        # Post-processing statistics
│   └── post_visualizations.py  # Post-processing visualizations
│
├── output/             # All outputs are saved here
│   ├── IQL/           # Self-play results for IQL
│   ├── MixedPlay/     # Mixed-play results
│   ├── Multiple/      # Multiple repetition results
│   └── Final/         # Final results (30 repetitions)
│
├── run.py              # Self-Play training script
├── run_mixed.py        # Mixed-Play training script
├── run_multiple.py     # Multiple repetitions script
├── train.py            # Core training functions
└── requirements.txt    # Python dependencies
```

## Output Locations

All outputs are saved in the `output/` directory:

- **Self-Play results**: `output/{ALGORITHM}/{runname}/`
  - Example: `output/IQL/10000eps_100epL_16dec2025_IQL/`

- **Mixed-Play results**: `output/MixedPlay/{runname}/`
  - Example: `output/MixedPlay/300eps_50epL_15dec2025_IQLAE_vs_IQLAE/`

- **Multiple repetition results**: 
  - `output/Multiple/{runname}/` (for < 30 repetitions)
  - `output/Final/{runname}/` (for 30 repetitions)

Each output directory typically contains:
- `eval_returns.csv`: Evaluation performance data
- `eval_image.png` or similar: Visualization plots
- `video/`: Directory with evaluation videos (if enabled)

## Environment Types

The following environment types are supported:

- `"cf"`: Custom foraging environment
- `"cf1f"`: Custom foraging with one food
- `"f"`: Standard foraging environment (lbforaging)
- `"m"`: Matrix game
- `"mc"`: Move chair game
- `"mcs"`: Move chair simple
- `"mcc"`: Move chair coordination

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **For Self-Play:**
   - Edit `CONFIG` in `run.py`
   - Run: `python run.py`

3. **For Mixed-Play:**
   - Edit `CONFIG` in `run_mixed.py`
   - Run: `python run_mixed.py`

4. **For Multiple Repetitions:**
   - Edit `CONFIG` in `run_multiple.py`
   - Run: `python run_multiple.py`

## Notes

- All scripts use configuration dictionaries (`CONFIG`) that can be modified directly in the script files
- Seeds can be set for reproducibility (set `"seed"` in CONFIG)
- Videos are automatically generated for foraging environments when `"visualise": True`
- Evaluation frequency and episodes can be adjusted in the CONFIG
