# Observational Learning with Gated Information: A Lightweight RL Simulation Testbed and Latent Signal Models

Code-first research repository for the social timing / observational learning experiments from "Observational Learning with Gated Information: A Lightweight RL Simulation Testbed and Latent Signal Models".

`main.py` is the canonical runnable implementation. The notebook under `notebooks/` is kept as archived exploratory analysis and should not be treated as the source of truth.

## Repository Layout

- `main.py`: unified CLI for training, comparison suites, latent-signal simulation, and LR sweeps
- `requirements.txt`: minimal runtime dependencies
- `notebooks/`: archived notebook history with outputs cleared
- `paper/`: paper PDF and manuscript assets
- `outputs/`: generated figures, reports, and experiment runs; ignored for version control

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## CLI Usage

Show all options:

```bash
python main.py --help
```

Run a default DQN experiment:

```bash
python main.py --algo dqn --episodes 800
```

Compare familiarity on/off:

```bash
python main.py --compare_suite --compare_familiarity --algo dqn
```

Compare learner profiles with empathy enabled:

```bash
python main.py --compare_suite --compare_profiles --empathy on --algo dqn
```

Run a lightweight smoke test:

```bash
python main.py --algo tabular --episodes 1 --no_plots --save_dir outputs/smoke_tabular
```

## What the CLI Covers

- DQN, PPO, and tabular Q-learning baselines
- Social visibility, familiarity, empathy, and teacher palatability manipulations
- Multi-seed comparison suites with mean +/- SEM plots
- Permutation-test significance reports
- Optional latent "DA-like" signal synthesis
- Learning-rate sweeps

## Outputs

By default, runs write to `outputs/`. Typical artifacts include:

- `comparison_suite_*.pdf`
- `learning_curves_*.pdf`
- `significance_report.txt`
- `lr_sweep_scores.txt`

## Citation

If you use this repository in research, cite:

```bibtex
@article{zhu2026observational,
  title={Observational Learning with Gated Information: A Lightweight RL Simulation Testbed and Latent Signal Models},
  author={Zhu, Chengyuan and Wu, Jialai},
  journal={research square},
  year={2026}
}
```

## Notes

- The notebook contains historical development snapshots, so it is intentionally preserved as reference material rather than polished production code.
