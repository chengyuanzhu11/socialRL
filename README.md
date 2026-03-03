# Observational Learning with Gated Information

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Supported-orange)

This repository contains the code for the lightweight Reinforcement Learning (RL) simulation testbed introduced in the paper: **"Observational Learning with Gated Information: A Lightweight RL Simulation Testbed and Latent Signal Models"**.

## Overview

Observational learning is often framed as active information sampling. This project provides a Partially Observable Markov Decision Process (POMDP) environment—`Social FoodEnv`—designed to study the **social-timing problem**. In this environment, an observer agent must decide when to actively spend time watching a demonstrator to reduce timing uncertainty about an upcoming, brief reward window, balancing the cost of observation against the need to act.

### Key Features

* **Gated Information Mechanism:** The agent is "social-blind" by default. Social timing features (demonstrator cues and remaining window estimates) are strictly gated and only accessible when the agent executes an explicit `Observe` action.
* **Imperfect Sensing:** Cue detection is probabilistic, and time estimates contain noise, simulating realistic sensory constraints.
* **Multi-Algorithm Baselines:** Includes ready-to-use implementations of:
  * Dueling Double Deep Q-Network (D3QN)
  * Proximal Policy Optimization (PPO) actor-critic
* **Controlled Condition Manipulations:** Built-in support for ablation studies, including:
  * *Social-blind*: Social channel completely disabled.
  * *Impaired Tracking*: Reduced detection reliability and noisier estimates.
  * *Familiarity*: A latent variable that improves effective sensing over repeated exposures.
  * *Teacher Palatability*: Manipulations of the demonstrator's cue-burst duration.
* **Latent Signal Synthesis:** A pipeline to extract reward-related internal variables (e.g., TD error/RPE, observation-driven value updates, action-initiation salience) for event-aligned analysis.

## Installation

We recommend using a virtual environment (e.g., Conda or venv).

```bash
git clone [https://github.com/chengyuanzhu11/](https://github.com/chengyuanzhu11/)<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt

```

*Main Dependencies: `torch`, `gymnasium`, `numpy`, `matplotlib`, `pandas*`

## Environment Details (State & Action Space)

* **Action Space (Discrete: 5):** `[Move Left, Stay, Move Right, Observe, Consume]`
* **State Space (Continuous: 6D/7D Vector):** 1. Normalized agent position
2. Normalized resource-site position
3. Gated cue indicator (Non-zero only on `Observe`)
4. Gated reward-window remaining estimate (Non-zero only on `Observe`)
5. Phase-to-next-nominal-bout (Context clock)
6. Normalized consume cooldown
7. *Optional:* Familiarity scalar $f \in [0,1]$

## Usage

### Training an Agent

To train the default Dueling Double DQN agent on the baseline environment:

```bash
python train.py --algo dqn --episodes 1200 --condition baseline

```

### Running Ablation Studies

You can easily switch conditions to replicate the paper's experiments:

```bash
# Train a social-blind observer
python train.py --algo dqn --condition social_blind

# Train with familiarity enabled
python train.py --algo dqn --condition familiarity_on

```

### Evaluating and Extracting Latent Signals

After training, run the evaluation script to generate behavioral metrics and synthesize peri-event latent signal traces (e.g., TD error aligned to cue-burst termination).

```bash
python evaluate.py --model_path checkpoints/dqn_baseline.pth --extract_signals

```

## Citation

If you use this testbed in your research, please cite our preprint:

```bibtex
@article{zhu2026observational,
  title={Observational Learning with Gated Information: A Lightweight RL Simulation Testbed and Latent Signal Models},
  author={Zhu, Chengyuan and Wu, Jialai},
  journal={bioRxiv},
  year={2026}
}
