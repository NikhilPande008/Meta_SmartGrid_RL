import os
import sys
import time
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Load local .env if present (won't affect hackathon environment)
load_dotenv()

def _ensure_src_on_path() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(repo_root, "src"))

def _make_physics(*, max_solar_kwh: float, demand_multiplier: float) -> Callable[[int], tuple[float, float]]:
    def physics(hour: int) -> tuple[float, float]:
        if 6 <= hour <= 18:
            solar = max_solar_kwh * float(np.sin(np.pi * (hour - 6) / 12))
        else:
            solar = 0.0
        demand_base = 2.0 * demand_multiplier
        if 18 <= hour <= 22:
            demand = demand_base + (6.0 * demand_multiplier) * float(np.sin(np.pi * (hour - 18) / 4))
        else:
            demand = demand_base
        return float(np.clip(solar, 0.0, max_solar_kwh)), float(np.clip(demand, 0.0, 8.0 * demand_multiplier))
    return physics

def _plot_energy_balance(hours, solar, demand, battery):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hours, solar, label="Solar (kWh)", linewidth=2)
    ax.plot(hours, demand, label="Demand (kWh)", linewidth=2)
    ax.plot(hours, battery, label="Battery (kWh)", linewidth=2, color='green')
    ax.set_xlabel("Hour")
    ax.set_ylabel("kWh")
    ax.set_title("SmartGrid Energy Balance (24h)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def main() -> None:
    st.set_page_config(page_title="Meta SmartGrid RL", layout="wide")
    st.title("🌿 Meta SmartGrid RL Dashboard")

    with st.sidebar:
        st.header("Settings")
        solar_mult = st.slider("Solar Multiplier", 0.0, 2.0, 1.0)
        demand_mult = st.slider("Demand Multiplier", 0.0, 2.0, 1.0)
        skip_llm = st.checkbox("Skip LLM Evaluation", value=False)
        run = st.button("Run Simulation", type="primary")

    left, right = st.columns([2, 1])
    chart_slot = left.empty()
    llm_score_slot = right.empty()
    llm_feedback_slot = right.empty()

    if run:
        try:
            _ensure_src_on_path()
            from meta_smartgrid_rl.env import SustainableGridEnv
            
            # Setup environment variables for the session
            if skip_llm:
                os.environ["SKIP_LLM"] = "1"
            
            env = SustainableGridEnv(max_solar_kwh=12.0 * solar_mult)
            env._get_physics_update = _make_physics(max_solar_kwh=12.0 * solar_mult, demand_multiplier=demand_mult)
            
            env.reset()
            h_list, s_list, d_list, b_list = [], [], [], []

            for _ in range(24):
                curr_h = int(env.state[3])
                action = 1 if 10 <= curr_h <= 15 else (2 if 18 <= curr_h <= 22 else 0)
                _, _, done, trunc, info = env.step(action)
                
                log = env.history[-1]
                h_list.append(log["hour"])
                s_list.append(log["solar"])
                d_list.append(log["demand"])
                b_list.append(log["battery_charge"])

                fig = _plot_energy_balance(h_list, s_list, d_list, b_list)
                chart_slot.pyplot(fig)
                plt.close(fig)
                if done or trunc: break
                time.sleep(0.02)

            llm_score_slot.success(f"LLM Score: {info.get('llm_score', 'N/A')}")
            llm_feedback_slot.text_area("Feedback", value=str(info.get('llm_feedback', 'No feedback available.')), height=200)

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()