import sys
import os
import numpy as np

# 1. SETUP PATHING
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 2. IMPORT ENVIRONMENT
try:
    from meta_smartgrid_rl.env import SustainableGridEnv
except ImportError:
    sys.path.append("/app")
    from meta_smartgrid_rl.env import SustainableGridEnv

def main():
    # Initialize the environment
    try:
        env = SustainableGridEnv()
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return

    # Configuration for the evaluation
    task_name = "smartgrid_balancing"
    num_episodes = 1 
    
    # 3. STRUCTURED OUTPUT LOOP
    # The validator looks for these exact strings in stdout
    print(f"[START] task={task_name}", flush=True)

    obs, info = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done:
        # AGENT LOGIC: 
        # Replace with your actual model prediction if you have a trained model
        # For now, using a simple heuristic or random action
        action = env.action_space.sample() 
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        # Print step details as required
        print(f"[STEP] step={steps} reward={reward:.4f}", flush=True)

        if steps > 1000: # Safety breakout
            break

    # 4. FINAL SCORE
    # The score should typically be a normalized value (0 to 1)
    # Adjust the 'score' calculation based on your env's max possible reward
    final_score = max(0.0, min(1.0, total_reward / 100.0)) 
    
    print(f"[END] task={task_name} score={final_score:.4f} steps={steps}", flush=True)

    # Clean exit
    sys.exit(0)

if __name__ == "__main__":
    main()