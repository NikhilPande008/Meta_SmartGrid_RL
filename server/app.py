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
        # Final fallback for flat directory structures
        try:
            from env import SustainableGridEnv
        except ImportError:
            pass

# 3. FASTAPI INSTANTIATION
# We define this globally so the platform can import it via 'inference:app'
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize env safely
try:
    env = SustainableGridEnv()
except Exception:
    env = None

# 4. ENDPOINTS
@app.get("/")
async def health():
    return {"status": "alive"}

@app.post("/reset")
async def reset():
    if env is None: return {"error": "Env not found"}
    obs, info = env.reset()
    return {"observation": obs.tolist() if hasattr(obs, 'tolist') else obs, "info": info}

@app.post("/step")
async def step(action_data: dict):
    if env is None: return {"error": "Env not found"}
    action = action_data.get("action", 0)
    obs, reward, done, truncated, info = env.step(action)
    return {
        "observation": obs.tolist() if hasattr(obs, 'tolist') else obs,
        "reward": float(reward),
        "done": bool(done or truncated),
        "info": info
    }

# 5. THE CRITICAL FIX FOR PHASE 2
def main():
    """
    The platform runs: 'python inference.py'
    The validator log shows the platform starts its own server on 7860.
    Therefore, this script must NOT run uvicorn.run().
    It must simply stay alive or exit gracefully.
    """
    import time
    print("Agent Process Started. Handing over control to validator server.")
    try:
        while True:
            time.sleep(10) # Minimal CPU usage, keeps process from exiting
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()