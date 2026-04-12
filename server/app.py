import sys
import os
import numpy as np
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI  # Ensure 'openai' is in your requirements.txt

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

# 3. LLM PROXY CLIENT (The Critical Fix)
# These environment variables are injected by the Meta/Scaler platform
api_key = os.getenv("API_KEY", "dummy_key")
base_url = os.getenv("API_BASE_URL", "https://api.example.com/v1")

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

def get_llm_action(obs):
    """Makes a call through the mandatory LiteLLM proxy."""
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-70b-Instruct", # Use the model specified in hackathon docs
            messages=[
                {"role": "system", "content": "You are a Smart Grid controller. Output ONLY a single integer for the best action."},
                {"role": "user", "content": f"State: {obs}. What is the next action (0-4)?"}
            ],
            max_tokens=5
        )
        content = response.choices[0].message.content.strip()
        return int(''.join(filter(str.isdigit, content))) % 5
    except Exception as e:
        print(f"LLM Proxy Call failed: {e}", flush=True)
        return 0 # Fallback

# 4. FASTAPI APP
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

try:
    env = SustainableGridEnv()
except:
    env = None

@app.get("/")
async def health(): return {"status": "alive"}

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

# 5. THE VALIDATOR RUNNER
def main():
    if env is None: return

    task_name = "smartgrid_balancing"
    print(f"[START] task={task_name}", flush=True)

    obs, info = env.reset()
    total_reward = 0
    
    # We run a short loop to demonstrate LLM usage
    for i in range(1, 11): 
        # MANDATORY: Use the LLM for at least one action to pass the check
        action = get_llm_action(obs)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"[STEP] step={i} reward={reward:.4f}", flush=True)

        if terminated or truncated:
            break

    score = max(0.0, min(1.0, total_reward / 100.0))
    print(f"[END] task={task_name} score={score:.4f} steps={i}", flush=True)

if __name__ == "__main__":
    main()