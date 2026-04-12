import sys
import os
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Pathing for subfolder
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from meta_smartgrid_rl.env import SustainableGridEnv
except ImportError:
    # Fallback for Docker
    sys.path.append("/app")
    from meta_smartgrid_rl.env import SustainableGridEnv

# The validator looks for this 'app' object to serve the API
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize environment
env = SustainableGridEnv()

@app.get("/")
async def health():
    return {"status": "alive", "message": "SmartGrid API is Running"}

@app.post("/reset")
async def reset():
    obs, info = env.reset()
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

def main():
    """
    Only starts uvicorn if explicitly called in a local environment
    without an existing server.
    """
    print("SmartGrid RL Inference Script Initialized.")
    
    # Check if we are running in the validation environment
    # Most hackathon platforms set an ENV variable like 'VALIDATION' or 'PORT'
    if os.getenv("KUBERNETES_SERVICE_HOST") is None and os.getenv("PHASE") is None:
        try:
            import uvicorn
            # Only run locally for debugging
            print("Local mode detected. Starting Uvicorn...")
            uvicorn.run(app, host="0.0.0.0", port=7860)
        except Exception as e:
            print(f"Server already running or error: {e}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()