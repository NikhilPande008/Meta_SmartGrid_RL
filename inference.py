import sys
import os
import numpy as np
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
        # Fallback if the folder is in the same directory
        try:
            from env import SustainableGridEnv
        except ImportError:
            pass

# 3. FASTAPI APP (Required by the "Internal" server)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
try:
    env = SustainableGridEnv()
except Exception:
    env = None

# Track metrics for the [END] block
stats = {"total_reward": 0.0, "steps": 0, "task": "smartgrid_balancing"}

@app.on_event("startup")
async def startup_event():
    print(f"[START] task={stats['task']}", flush=True)

@app.get("/")
async def health():
    return {"status": "alive"}

@app.post("/reset")
async def reset():
    if env is None: return {"error": "Env not found"}
    obs, info = env.reset()
    # Reset local stats for a new run
    stats["total_reward"] = 0.0
    stats["steps"] = 0
    return {
        "observation": obs.tolist() if hasattr(obs, 'tolist') else obs,
        "info": info
    }

@app.post("/step")
async def step(action_data: dict):
    if env is None: return {"error": "Env not found"}
    
    action = action_data.get("action", 0)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Update Stats
    stats["steps"] += 1
    stats["total_reward"] += reward
    done = terminated or truncated

    # REQUIRED STRUCTURED LOGGING
    print(f"[STEP] step={stats['steps']} reward={reward:.4f}", flush=True)
    
    if done:
        score = max(0.0, min(1.0, stats["total_reward"] / 100.0))
        print(f"[END] task={stats['task']} score={score:.4f} steps={stats['steps']}", flush=True)

    return {
        "observation": obs.tolist() if hasattr(obs, 'tolist') else obs,
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }

# 4. MAIN FUNCTION (Required by validator process)
def main():
    """
    The script must exist and be callable, but since the platform 
    is running uvicorn on 'inference:app', we just print a readiness signal.
    """
    print("Inference module loaded. App attribute exported.", flush=True)

if __name__ == "__main__":
    main()