# Autonomous Highway Driving with Deep Q-Learning

An intelligent reinforcement learning agent that learns to navigate multi-lane highway environments safely and efficiently. This project implements Deep Q-Network (DQN) training with performance benchmarking against traditional heuristic baselines and human-level controls.

## Project Overview

This repository contains a complete implementation of autonomous highway driving using deep reinforcement learning. The agent learns to make optimal navigation decisions in complex traffic scenarios, balancing two critical objectives: maximizing velocity to maintain traffic flow and maintaining safe distances to prevent collisions.

The core approach utilizes a Deep Q-Network architecture trained with experience replay and target networks. The agent observes a discretized state representation of the highway environment (including ego vehicle metrics and surrounding traffic) and selects from five discrete actions: change lanes left, remain idle, change lanes right, accelerate, or decelerate.

## Performance Metrics

The trained DQN agent demonstrates significant improvements over traditional rule-based approaches:

| Agent | Mean Reward | Crash Rate |
|-------|-------------|------------|
| Heuristic Baseline | 8.78 | 90% |
| **DQN** | **27.54** | **15%** |
| Human Control | 33.03 | 5% |

The DQN agent achieves approximately 83% of human-level performance while substantially outperforming the heuristic baseline, reducing collision rates by 75 percentage points.

## Methodology

### Architecture

The DQN agent consists of:
- **Q-Network**: A fully connected neural network with two hidden layers (256 units each) that maps state observations to Q-values for each action
- **Target Network**: A separate network updated periodically to provide stable target values
- **Experience Replay Buffer**: Stores 10,000 transitions for random sampling during training

### Training Process

#### Deep Q-Learning Algorithm

The agent learns an action-value function $Q(s, a)$ that estimates the expected future reward when taking action $a$ in state $s$. The Q-values are updated using the Bellman equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

where $\alpha$ is the learning rate, $\gamma$ is the discount factor, $r$ is the immediate reward, and $s'$ is the next state.

#### Loss Function

For each mini-batch of transitions, the network is trained by minimizing the Mean Squared Error:

$$L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2]$$

where $\theta$ are the current network parameters and $\theta^-$ are the target network parameters updated periodically.

#### Exploration Strategy

The agent uses epsilon-greedy exploration:

$$a = \begin{cases} 
\text{random action} & \text{with probability } \epsilon \\
\arg\max_a Q(s, a) & \text{with probability } 1 - \epsilon
\end{cases}$$

The exploration rate decays as: $\epsilon_t = \max(\epsilon_{\min}, \epsilon_0 \cdot \gamma_\epsilon^t)$

#### Training Hyperparameters

- **Environment**: Highway-Fast environment with 50 vehicles at 40ms decision intervals
- **Training Steps**: 20,000 environment interactions
- **Discount Factor**: $\gamma = 0.99$
- **Learning Rate**: $\alpha = 0.001$ (Adam optimizer)
- **Batch Size**: 32 transitions
- **Exploration**: $\epsilon$ decays from 1.0 to 0.01

## Key Features

- Full DQN implementation with experience replay and target networks
- Modular agent architecture for extensibility
- Comprehensive evaluation suite comparing multiple agent types
- Pre-trained model weights for immediate inference
- Heuristic baseline for reference performance
- Manual control interface for human-level testing

## Installation

### Requirements

- Python 3.12+
- PyTorch (GPU-accelerated)
- Gymnasium and Highway-Env environments
- NumPy, Matplotlib, Pandas

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/ricca200xx/autonomous-highway-dqn.git
   cd autonomous-highway-dqn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training a New Agent

To train the DQN agent from scratch:

```bash
python training.py
```

This will:
- Initialize the DQN agent with a fresh neural network
- Train for 20,000 environment steps
- Save the trained weights to `model_weights.pth`
- Display periodic training progress and statistics

### Evaluating Trained Models

To evaluate the trained DQN agent against the heuristic baseline:

```bash
python evaluate.py
```

This runs both agents for 20 episodes each and reports:
- Average cumulative reward
- Collision rates
- Episode length statistics

### Manual Control Testing

To test the environment with manual keyboard control:

```bash
python manual_control.py
```

Use arrow keys or WASD to control the vehicle and understand the environment dynamics.

## Project Structure

```
.
├── training.py          # DQN agent training script
├── evaluate.py          # Evaluation and comparison script
├── manual_control.py    # Interactive manual control interface
├── your_baseline.py     # Heuristic baseline agent implementation
├── model_weights.pth    # Pre-trained DQN model weights
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Technical Details

### State Representation

The agent observes a 25-dimensional state vector $s \in \mathbb{R}^{25}$ structured as a $5 \times 5$ matrix:

$$s = \begin{bmatrix}
\text{ego presence} & \text{ego } x & \text{ego lane} & \text{ego } v_x & \text{ego } v_y \\
\text{veh}_1 \text{ presence} & \text{veh}_1 x & \text{veh}_1 \text{ lane} & \text{veh}_1 v_x & \text{veh}_1 v_y \\
\vdots & \vdots & \vdots & \vdots & \vdots \\
\text{veh}_4 \text{ presence} & \text{veh}_4 x & \text{veh}_4 \text{ lane} & \text{veh}_4 v_x & \text{veh}_4 v_y
\end{bmatrix}$$

where presence is binary (0 or 1), positions and velocities are normalized to $[-1, 1]$ for stable learning.

### Action Space

The discrete action space consists of 5 actions, forming the set $\mathcal{A} = \{0, 1, 2, 3, 4\}$:

| Action ID | Operation |
|-----------|-----------|
| 0 | Change lane left |
| 1 | Idle (maintain state) |
| 2 | Change lane right |
| 3 | Accelerate |
| 4 | Decelerate |

### Reward Function

The reward signal $r(s, a, s')$ is designed to incentivize safe and efficient driving:

$$r(s, a, s') = \begin{cases}
r_{\text{velocity}} & \text{if } v > 0 \text{ and no collision} \\
r_{\text{collision}} & \text{if collision occurs} \\
r_{\text{velocity}} - r_{\text{lane\_change}} & \text{if lane change action is executed}
\end{cases}$$

where:
- $r_{\text{velocity}} > 0$ provides positive feedback for maintaining speed
- $r_{\text{collision}} \ll 0$ is a large penalty (typically -1) for crashes
- $r_{\text{lane\_change}} > 0$ is a small penalty for unnecessary lane changes

## Baseline Comparison

### Heuristic Agent

A hand-crafted rule-based agent using:
- Safety distance thresholds (0.15 and 0.3 units)
- Hierarchical decision logic for lane changes
- Speed limit enforcement (90% of max speed)
- Collision avoidance through reactive steering

### Human Control

Provides the upper-bound performance baseline, demonstrating optimal decision-making in the environment.

## Results Discussion

The DQN agent significantly outperforms the heuristic rules-based approach by learning sophisticated decision patterns from raw environment observations.

### Performance Analysis

Let $R_{\text{DQN}}$, $R_{\text{heuristic}}$, and $R_{\text{human}}$ denote the mean episode rewards for DQN, heuristic, and human agents respectively:

- DQN improvement over heuristic: $\frac{R_{\text{DQN}} - R_{\text{heuristic}}}{R_{\text{heuristic}}} = \frac{27.54 - 8.78}{8.78} \approx 213\%$

- DQN relative to human performance: $\frac{R_{\text{DQN}}}{R_{\text{human}}} = \frac{27.54}{33.03} \approx 83\%$

- Crash rate improvement: The DQN reduces collision probability to $P_{\text{crash, DQN}} = 0.15$ compared to the heuristic baseline $P_{\text{crash, heuristic}} = 0.90$, representing a reduction factor of $\frac{0.90}{0.15} = 6.0\times$

### Key Advantages

- Adaptive behavior based on traffic patterns
- Learned trade-offs between speed and safety
- Generalization to unseen traffic configurations
- Stable performance across multiple evaluation episodes

The substantial crash rate reduction and reward improvement demonstrate the effectiveness of learning-based control in safety-critical driving scenarios. The agent's performance approaching human-level ($83\%$) validates the DQN approach despite its simplicity compared to more advanced algorithms.

## Technologies & Dependencies

- **PyTorch**: Deep learning framework for neural networks
- **Gymnasium**: Standardized environment interface
- **Highway-Env**: Multi-lane highway driving simulation
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and statistics

See [requirements.txt](requirements.txt) for complete version specifications.

## Future Directions

Potential extensions to this project:
- Dueling DQN architecture for improved value estimation
- Double DQN to reduce overestimation bias
- Prioritized experience replay for efficient learning
- Continuous control with policy gradient methods
- Multi-agent scenarios with cooperative training
- Real-world simulation transfer learning

## Authors

Riccardo Niccolo Agosti
