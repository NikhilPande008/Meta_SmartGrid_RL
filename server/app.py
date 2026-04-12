import sys
import os
import uvicorn
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# FIX: Pathing for subfolder. We need to look one level UP to find meta_smartgrid_rl
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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SustainableGridEnv()

@app.get("/")
async def health():
    return {"status": "alive", "message": "SmartGrid API is Running"}

@app.post("/")
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

# FIX: Added explicit main() function as requested by the validator
def main():
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(
        "server.app:app", 
        host="0.0.0.0", 
        port=port, 
        proxy_headers=True, 
        forwarded_allow_ips="*"
    )

# FIX: Standard boilerplate to make the function callable
if __name__ == "__main__":
    main()