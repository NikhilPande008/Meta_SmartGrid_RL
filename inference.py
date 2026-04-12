import sys
import os
import numpy as np
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 1. SETUP PATHING
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 2. IMPORT ENVIRONMENT
try:
    from meta_smartgrid_rl.env import SustainableGridEnv
except ImportError:
    sys.path.append("/app")
    try:
        from meta_smartgrid_rl.env import SustainableGridEnv
    except ImportError:
        try:
            from env import SustainableGridEnv
        except ImportError:
            pass

# 3. FASTAPI INSTANCE (Exporting 'app' for the Internal Server)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize env globally
try:
    env = SustainableGridEnv()
except:
    env = None

# API ENDPOINTS
@app.get("/")
async def health():
    return {"status": "alive"}

@app.post("/reset")
async def reset():
    obs, info = env.reset()
    return {"observation": obs.tolist() if hasattr(obs, 'tolist') else obs, "info": info}

@app.post("/step")
async def step(action_data: dict):
    action = action_data.get("action", 0)
    obs, reward, terminated, truncated, info = env.step(action)
    return {
        "observation": obs.tolist() if hasattr(obs, 'tolist') else obs,
        "reward": float(reward),
        "done": bool(terminated or truncated),
        "info": info
    }

# 4. THE VALIDATOR RUNNER (The fix for "No structured output")
def main():
    """
    This function runs the actual evaluation loop and prints the 
    exact strings the validator's parser is looking for.
    """
    if env is None:
        print("Environment not found, exiting.")
        return

    task_name = "smartgrid_balancing"
    
    # START BLOCK
    print(f"[START] task={task_name}", flush=True)

    obs, info = env.reset()
    total_reward = 0
    steps = 0
    max_steps = 100 # Adjust based on hackathon requirements

    for i in range(1, max_steps + 1):
        # Sample an action (Replace with your model prediction if available)
        action = env.action_space.sample() 
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps = i
        
        # STEP BLOCK
        print(f"[STEP] step={steps} reward={reward:.4f}", flush=True)

        if terminated or truncated:
            break

    # END BLOCK
    # Normalizing score: Adjust the divisor (100.0) to match your env's scale
    score = max(0.0, min(1.0, total_reward / 100.0))
    print(f"[END] task={task_name} score={score:.4f} steps={steps}", flush=True)

# 5. EXECUTION LOGIC
if __name__ == "__main__":
    # If the platform runs 'python inference.py', this executes the loop
    # If the platform runs 'uvicorn inference:app', this is ignored, but 'app' is still exported
    main()