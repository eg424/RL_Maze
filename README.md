# Reinforcement Learning in Maze Navigation

This repository contains coursework for **BIOE70077 - Reinforcement Learning for Bioengineers** at Imperial College London. It compares the implementation of both **Dynamic Programming** and **Monte Carlo** methods in a maze environment simulating a drug delivery problem in bioengineering.

## Project Overview

The task was to train an agent to navigate a grid-like maze with obstacles and absorbing states representing biological barriers and drug targets. The project explored:

- **Dynamic Programming (Policy Iteration)**
- **Monte Carlo Learning (First-Visit, with Decaying ε-Greedy Policy)**
- **Exploration Strategy Comparison (Fixed ε, Decaying ε, and SoftMax)**
- **Sensitivity Analysis**

The environment is inspired by drug delivery systems, with RL techniques used to discover optimal paths under probabilistic movement constraints.

## Algorithms Implemented

### Dynamic Programming Agent
- **Approach:** Policy Iteration
- **Assumptions:** Full knowledge of transition matrix (T) and reward function (R)
- **Highlights:**
  - Converges quickly in small state space
  - Value function and policy grid visualizations
  - Explored the effects of different γ and transition probabilities

### Monte Carlo Agent
- **Approach:** First-Visit Monte Carlo with Decaying ε-Greedy
- **Assumptions:** Model-free learning using sampled episodes
- **Highlights:**
  - Optimistic Q-value initialization
  - Decaying learning rate and exploration rate
  - Learning curve across 400 episodes
  - Policy and value function visualizations

## Exploration Strategies Compared

| Strategy         | Description                                        | Notes |
|------------------|----------------------------------------------------|-------|
| Fixed ε-Greedy   | Constant exploration rate                          | High variance, did not converge efficiently |
| Decaying ε-Greedy| Exploration rate decays over time                 | Balanced convergence and exploration |
| SoftMax          | Action probability ∝ Q-values (temperature-based) | Best stability and learning performance |

Each method was evaluated by comparing learning curves and policy convergence.

## Results

- **SoftMax strategy** yielded the highest total rewards and most stable convergence.
- The **DP agent** converged faster due to full environment access.
- **Monte Carlo** handled uncertainty well, especially with exploration strategy tuning.

## Key Takeaways

- **Policy Iteration** is efficient in small, fully-known MDPs.
- **Model-free agents** (like Monte Carlo) benefit from smart exploration strategies.
- **Exploration-exploitation tradeoff** is crucial in noisy environments.
- **SoftMax** performed best, balancing fast learning and policy optimality.
