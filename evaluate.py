import gymnasium
import highway_env
import numpy as np
import torch
import torch.nn as nn
import random
import json
from your_baseline import HeuristicAgent


# set the seed 
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN
class DQNNetwork(nn.Module):
    def __init__(self, state_size=25, action_size=5):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# dqna agent
class DQNAgent:
    def __init__(self, state_size=25, action_size=5):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = DQNNetwork(state_size, action_size).to(device)
    
    def select_action(self, state, training=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()
    
    def load(self, path='model_weights.pth'):
        self.q_network.load_state_dict(torch.load(path, map_location=device))
        self.q_network.eval()

#evaluation
def evaluate_agents(num_eval_episodes=20):
    env_name = "highway-v0"
    
    env = gymnasium.make(env_name,
                         config={'action': {'type': 'DiscreteMetaAction'}, "lanes_count": 3, "ego_spacing": 1.5},
                         render_mode=None)
    
    heuristic_agent = HeuristicAgent()
    dqn_agent = DQNAgent(state_size=25, action_size=5)
    dqn_agent.load('model_weights.pth')
    print("Agents loaded successfully!\n")
    
    heuristic_rewards = []
    heuristic_crashes = []
    dqn_rewards = []
    dqn_crashes = []
    

    #heuristic agent evaluation
    print("evaluating heuristic agent")
    for episode in range(1, num_eval_episodes + 1):
        state, _ = env.reset(seed=episode)
        state = state.reshape(-1)
        done, truncated = False, False
        
        episode_steps = 0
        episode_return = 0
        crashed = False
        
        while not (done or truncated):
            episode_steps += 1
            action = heuristic_agent.act(state)
            
            state, reward, done, truncated, _ = env.step(action)
            state = state.reshape(-1)
            
            episode_return += reward
            
            if done:  
                crashed = True
        
        heuristic_rewards.append(episode_return)
        heuristic_crashes.append(1 if crashed else 0)
        print(f"Episode {episode:2d} | Return: {episode_return:7.2f} | Steps: {episode_steps:3d} | Crash: {crashed}")
    
    # dqn agent evaluation
    print("evaluating dqn agent")
    for episode in range(1, num_eval_episodes + 1):
        state, _ = env.reset(seed=episode)
        state = state.reshape(-1)
        done, truncated = False, False
        
        episode_steps = 0
        episode_return = 0
        crashed = False
        
        while not (done or truncated):
            episode_steps += 1
            action = dqn_agent.select_action(state, training=False)
            
            state, reward, done, truncated, _ = env.step(action)
            state = state.reshape(-1)
            
            episode_return += reward
            
            if done:  
                crashed = True
        
        dqn_rewards.append(episode_return)
        dqn_crashes.append(1 if crashed else 0)
        print(f"Episode {episode:2d} | Return: {episode_return:7.2f} | Steps: {episode_steps:3d} | Crash: {crashed}")
    
    env.close()
    
    # statistics
    heuristic_mean = np.mean(heuristic_rewards)
    heuristic_std = np.std(heuristic_rewards)
    heuristic_crash_rate = np.mean(heuristic_crashes)
    
    dqn_mean = np.mean(dqn_rewards)
    dqn_std = np.std(dqn_rewards)
    dqn_crash_rate = np.mean(dqn_crashes)
    
    results = {
        'heuristic_rewards': np.array(heuristic_rewards),
        'heuristic_crashes': np.array(heuristic_crashes),
        'heuristic_mean': heuristic_mean,
        'heuristic_std': heuristic_std,
        'heuristic_crash_rate': heuristic_crash_rate,
        'dqn_rewards': np.array(dqn_rewards),
        'dqn_crashes': np.array(dqn_crashes),
        'dqn_mean': dqn_mean,
        'dqn_std': dqn_std,
        'dqn_crash_rate': dqn_crash_rate,
    }
    

    print(f"{'Agent':<20} {'Mean Reward':<15} {'Std Dev':<15} {'Crash Rate':<15}")
    print(f"{'Heuristic':<20} {heuristic_mean:>10.3f}     {heuristic_std:>10.3f}     {heuristic_crash_rate:>10.1%}")
    print(f"{'DQN Agent':<20} {dqn_mean:>10.3f}     {dqn_std:>10.3f}     {dqn_crash_rate:>10.1%}")

    
    # resul json saving
    results_data = {
        'heuristic_rewards': heuristic_rewards,
        'heuristic_crashes': heuristic_crashes,
        'heuristic_mean': float(heuristic_mean),
        'heuristic_std': float(heuristic_std),
        'heuristic_crash_rate': float(heuristic_crash_rate),
        'dqn_rewards': dqn_rewards,
        'dqn_crashes': dqn_crashes,
        'dqn_mean': float(dqn_mean),
        'dqn_std': float(dqn_std),
        'dqn_crash_rate': float(dqn_crash_rate),
    }
    
    with open('evaluate_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print("\nResults saved to: evaluate_results.json\n")


if __name__ == "__main__":
    evaluate_agents(num_eval_episodes=20)


