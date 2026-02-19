import gymnasium
import highway_env
import numpy as np
import time
import json

# manual policy
def evaluate_manual_policy(num_eval_episodes=20):
    env_name = "highway-v0"
    
    env = gymnasium.make(env_name,
                         config={"manual_control": True, "lanes_count": 3, "ego_spacing": 1.5},
                         render_mode='human')
    
    manual_rewards = []
    manual_crashes = []

    
    for episode in range(1, num_eval_episodes + 1):
        state, info = env.reset(seed=episode)
        done, truncated = False, False
        
        episode_steps = 0
        episode_return = 0
        crashed = False
        
        while not (done or truncated):
            episode_steps += 1
            action = env.action_space.sample()
            
            state, reward, done, truncated, info = env.step(action)
            
            try:
                env.render()
            except:
                pass  
            
            time.sleep(0.05)  
            
            episode_return += reward
            
            if done:
                crashed = True
        
        manual_rewards.append(episode_return)
        manual_crashes.append(1 if crashed else 0)
        print(f"Episode {episode:2d} | Return: {episode_return:7.2f} | Steps: {episode_steps:3d} | Crash: {crashed}")
    
    env.close()
    
    return np.array(manual_rewards), np.array(manual_crashes)

#execution
if __name__ == "__main__":
    rewards, crashes = evaluate_manual_policy(num_eval_episodes=20)
    
    print(f"Mean Reward: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Crash Rate: {np.mean(crashes):.1%}")
    
    # save json
    manual_mean = float(np.mean(rewards))
    manual_std = float(np.std(rewards))
    manual_crash_rate = float(np.mean(crashes))
    
    results_data = {
        'manual_rewards': rewards.tolist(),
        'manual_crashes': crashes.tolist(),
        'manual_mean': manual_mean,
        'manual_std': manual_std,
        'manual_crash_rate': manual_crash_rate,
    }
    
    with open('manual_policy_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print("\nResults saved to: manual_policy_results.json")

