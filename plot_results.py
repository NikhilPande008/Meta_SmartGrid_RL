import os
import sys

import matplotlib.pyplot as plt


def main() -> None:
    # Allow running without installing the package
    repo_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(repo_root, "src"))

    from meta_smartgrid_rl.env import SustainableGridEnv

    env = SustainableGridEnv()
    env.reset()

    # Run a simple policy for 24 hours:
    # - charge when solar > demand
    # - discharge when demand > solar
    # - otherwise hold
    for _ in range(24):
        solar, demand, battery, hour = env.state  # raw state (not normalized)

        if solar > demand:
            action = 1  # charge
        elif demand > solar:
            action = 2  # discharge
        else:
            action = 0  # hold

        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    hours = [int(h["hour"]) for h in env.history]
    solar_curve = [float(h["solar"]) for h in env.history]
    demand_curve = [float(h["demand"]) for h in env.history]
    battery_curve = [float(h["battery_charge"]) for h in env.history]

    plt.figure(figsize=(12, 6))
    plt.plot(hours, solar_curve, label="Solar (kW)", linewidth=2)
    plt.plot(hours, demand_curve, label="Demand (kW)", linewidth=2)
    plt.plot(hours, battery_curve, label="Battery Level (kWh)", linewidth=2)

    plt.title("24-Hour Smart Grid Simulation")
    plt.xlabel("Hour")
    plt.ylabel("Power / Energy")
    plt.xticks(range(0, 24, 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(repo_root, "results_plot.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

