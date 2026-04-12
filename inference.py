import sys
import os
import uvicorn
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure root is in path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from meta_smartgrid_rl.env import SustainableGridEnv
except ImportError:
    sys.path.append("/app")
    from meta_smartgrid_rl.env import SustainableGridEnv

app = FastAPI()

# Add CORS so the validator doesn't get blocked
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

if __name__ == "__main__":
    # 7860 is the mandatory Hugging Face port
    port = int(os.getenv("PORT", 7860))
    # proxy_headers=True is critical for passing through the HF load balancer
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        proxy_headers=True, 
        forwarded_allow_ips="*"
    )