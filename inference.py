import os
import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel
from meta_smartgrid_rl.env import SustainableGridEnv

# --- Initialize App and Env ---
app = FastAPI(title="Meta SmartGrid OpenEnv API")
env = SustainableGridEnv()

class ActionRequest(BaseModel):
    action: int

@app.get("/")
async def health_check():
    return {"status": "active", "model": os.getenv("MODEL_NAME", "meta-smartgrid-rl")}

@app.post("/reset")
async def reset():
    """Handles the openenv_reset_post validation step."""
    obs, info = env.reset()
    # Convert numpy arrays to lists for JSON serialization
    return {
        "status": "success",
        "observation": obs.tolist() if hasattr(obs, 'tolist') else obs,
        "info": info
    }

@app.post("/step")
async def step(data: ActionRequest):
    """Executes a step in the environment."""
    obs, reward, terminated, truncated, info = env.step(data.action)
    return {
        "observation": obs.tolist() if hasattr(obs, 'tolist') else obs,
        "reward": float(reward),
        "done": bool(terminated or truncated),
        "info": info
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000)) 
    if os.getenv("SPACE_ID"): # Detects if running on Hugging Face
        port = 7860
    
    uvicorn.run(app, host="0.0.0.0", port=port)