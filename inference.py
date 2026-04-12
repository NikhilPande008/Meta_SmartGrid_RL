import sys
import os
import uvicorn
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Ensure Python can find your custom environment
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(repo_root, "src"))

try:
    from meta_smartgrid_rl.env import SustainableGridEnv
except ImportError as e:
    print(f"Error: Could not find meta_smartgrid_rl. Ensure it is in the 'src' folder. {e}")
    sys.exit(1)

app = FastAPI()
env = SustainableGridEnv()

@app.post("/")
@app.post("/reset")
async def reset():
    obs, info = env.reset()
    # Convert numpy array to list for JSON serialization
    return {
        "observation": obs.tolist() if isinstance(obs, np.ndarray) else obs,
        "info": info
    }

@app.post("/step")
async def step(action_data: dict):
    action = action_data.get("action", 0)
    obs, reward, done, truncated, info = env.step(action)
    return {
        "observation": obs.tolist() if isinstance(obs, np.ndarray) else obs,
        "reward": float(reward),
        "done": bool(done or truncated),
        "info": info
    }

if __name__ == "__main__":
    # Use port 7860 for Hugging Face compatibility, 8000 for local
    port = int(os.getenv("PORT", 7860 if os.getenv("SPACE_ID") else 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)