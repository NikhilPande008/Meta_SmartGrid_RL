import os
import sys


def main() -> None:
    # Allow running without installing the package
    repo_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(repo_root, "src"))

    from meta_smartgrid_rl.env import SustainableGridEnv

    env = SustainableGridEnv()
    obs, info = env.reset()

    total_reward = 0.0
    total_grid_import = 0.0

    for _ in range(24):
        # Simple time-of-use heuristic (matches env action meanings):
        # - 1: charge during late-morning/early-afternoon
        # - 2: discharge during evening peak
        # - 0: hold otherwise
        current_hour = int(env.state[3])
        if 10 <= current_hour <= 15:
            action = 1  # charge
        elif 17 <= current_hour <= 21:
            action = 2  # discharge
        else:
            action = 0  # hold
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        total_grid_import += float(info.get("grid_import", 0.0))

        if terminated or truncated:
            break

    print(f"Total reward (episode): {total_reward:.3f}")
    print(f"Total grid import (episode): {total_grid_import:.3f}")
    print(f"LLM score: {info.get('llm_score')}")
    print(f"LLM feedback: {info.get('llm_feedback')}")


if __name__ == "__main__":
    main()

