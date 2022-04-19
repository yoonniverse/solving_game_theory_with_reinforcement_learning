# Solving Games from Game Theory with Reinforcement Learning

This is the source code for solving some basic games from game theory with reinforcement learning. Reinforcement learning algorithm is based on PPO(https://arxiv.org/abs/1707.06347).

These are the games.
1. Prisoner’s Dilemma
2. Quality Choice Game
3. Goldenballs Game
4. Stag Hunt
5. Hawk-Dove
6. Penalty Kick Game
7. Kitty Genovese
8. Entry Deterrence Game
9. Ultimatum Game
10. Vote Buying
11. Committee Decision Making
12. Repeated Prisoner’s Dilemma

## Getting Started

1. `conda create -n gtrl python=3.8`
2. `conda activate gtrl`
3. `pip install torch matplotlib tqdm scipy`
4. `python run.py`

Then you can reproduce plots for training agent with multiple seeds for each game at `results/`.