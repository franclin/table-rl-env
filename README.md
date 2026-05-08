# Table RL Environment

This is an extension of a related project (www.weddingseatingplanner.com), in this case I am using RL whereas in the original version I used Google OR-tools.
A custom Gymnasium environment for solving constrained table arrangement problems using Reinforcement Learning.

## Problem Statement

Arrange N guests across M tables with:
- **Must-sit-together pairs** (guests who should share a table)
- **Avoid pairs** (guests who cannot sit together)
- **Capacity constraints** (max guests per table)

## Results

| Metric | Value |
|--------|-------|
| Success Rate | **100%** (100/100 episodes) |
| Avg. Steps to Solution | **3 steps** |
| Max Reward | 700/700 |

## Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/table-rl-env
cd table-rl-env
pip install -r requirements.txt

# Train the agent
python -m agents.train_stable

# Evaluate the trained model
python -m agents.evaluate