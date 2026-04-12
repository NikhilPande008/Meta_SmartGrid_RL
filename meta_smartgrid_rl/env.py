import numpy as np
import gymnasium as gym

from meta_smartgrid_rl.llm_scorer import GridCritic

class SustainableGridEnv(gym.Env):
    """
    Sustainable Grid Environment for RL.
    - solar_production: (0-100 kW)
    - home_demand: (0-100 kW)
    - battery_charge: (0-100 kWh)
    """
    def __init__(
        self,
        *,
        max_solar_kwh: float = 12.0,
        battery_capacity_kwh: float = 50.0,
        max_transfer_kwh: float = 10.0,
    ) -> None:
        super().__init__()
        # State: [solar_production, home_demand, battery_charge, hour]
        self.state = None
        self.hour = 0  # Time of day (0-23)
        self.max_kw = float(max_solar_kwh)
        self.max_kwh = float(battery_capacity_kwh)
        self.time_horizon = 24
        self.max_transfer = float(max_transfer_kwh)  # kWh per 1h step that can go in/out of battery

        # Battery limits
        self.battery_min = 0
        self.battery_max = self.max_kwh

        # Normalized observation space: [solar_norm, demand_norm, battery_norm, hour_norm] all in [0, 1].
        low = np.zeros(4, dtype=np.float32)
        high = np.ones(4, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space (discrete): 0=hold, 1=charge, 2=discharge
        self.action_space = gym.spaces.Discrete(3)

        self.history: list[dict[str, float]] = []
        self.critic = GridCritic(
            battery_capacity_kwh=self.max_kwh,
            max_solar_potential_kwh=self.max_kw,
        )

        self.reset()

    def _get_physics_update(self, hour: int) -> tuple[float, float]:
        """
        Simulate solar and demand using sin curves for base values.
        - Solar: Peaks at 12 (noon)
        - Demand: Baseline ~2 kWh, peaks ~8 kWh around evening (6pm-10pm).
        Values are in kWh per 1-hour step.
        """
        # Solar: sin from 6 to 18, peak at 12
        if 6 <= hour <= 18:
            # Sinusoid peak at 12, 0 at 6 & 18
            solar = self.max_kw * np.sin(np.pi * (hour - 6) / 12)
        else:
            solar = 0.0

        # Demand: baseline 2.0, evening peak adds up to +6.0 (total peak ~8.0).
        demand_base = 2.0
        if 18 <= hour <= 22:
            # Peak at 20, 0 at 18 & 22 (4-hour window)
            demand = demand_base + 6.0 * np.sin(np.pi * (hour - 18) / 4)
        else:
            demand = demand_base
        # Clamp values
        solar = float(np.clip(solar, 0.0, float(self.max_kw)))
        demand = float(np.clip(demand, 0.0, 8.0))
        return solar, demand

    def _get_obs(self) -> np.ndarray:
        """Build the normalized observation vector in [0, 1]."""
        solar, demand, battery_charge, hour = self.state
        solar_norm = float(solar) / float(self.max_kw)
        demand_norm = float(demand) / float(self.max_kw)
        battery_norm = float(battery_charge) / float(self.battery_max)
        hour_norm = float(hour) / float(self.time_horizon - 1)
        obs = np.array([solar_norm, demand_norm, battery_norm, hour_norm], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset the environment to the start of a new 24-hour episode.

        The environment simulates a single household with solar production, demand,
        and a battery. The episode always starts at hour 0.

        Args:
            seed: Optional RNG seed (Gymnasium API).
            options: Optional reset options (unused).

        Returns:
            A tuple of (observation, info). The observation is normalized to [0, 1]:
            - solar_norm = solar_kW / max_kw
            - demand_norm = demand_kW / max_kw
            - battery_norm = battery_kWh / battery_max
            - hour_norm = hour / (time_horizon - 1)
        """
        super().reset(seed=seed)
        self.hour = 0
        solar, demand = self._get_physics_update(self.hour)
        battery_charge = np.random.uniform(40, 60)
        self.state = np.array([solar, demand, battery_charge, self.hour], dtype=np.float32)
        self.history = []
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the simulation by one hour.

        Action semantics (discrete):
            - 0: hold (no intentional battery transfer)
            - 1: charge (only meaningful when solar > 0 / surplus exists)
            - 2: discharge

        Physics / energy balance:
            Let `actual_transfer` be the realized change in battery charge (kWh over 1 hour).
            Positive means charging (consumes energy), negative means discharging (supplies energy).

            Grid import is computed as:

                grid_import_kW = max(0, demand_kW + actual_transfer_kWh - solar_kW)

            (With a 1-hour step, kWh transfer is numerically aligned with kW over the step.)

        Reward:
            The agent is penalized for importing from the grid:

                reward = -grid_import_kW

            (No additional shaping term is applied to keep the objective simple.)

        Args:
            action: Discrete action in {0, 1, 2}.

        Returns:
            observation: Normalized observation in [0, 1].
            reward: Scalar float reward.
            terminated: Always False (no terminal state besides time truncation).
            truncated: True when the 24-hour cycle ends (hour wraps back to 0).
            info: Debug information including `grid_import`, and on truncation,
                `llm_score` and `llm_feedback`.
        """
        # 1. Validate action
        a = int(action)
        assert 0 <= a <= 2, "Action must be within valid range [0, 1, 2]"

        # 2. Unpack current raw state
        solar, demand, battery_old, current_hour = self.state
        self.hour = int(current_hour)

        # 3. Solar serves demand first; any deficit can be covered by the battery before grid import.
        solar_used_for_demand = min(float(solar), float(demand))
        remaining_demand = float(demand) - solar_used_for_demand  # >= 0
        solar_surplus = float(solar) - solar_used_for_demand  # >= 0

        # 4. Battery behavior (kWh per 1h step)
        # Charge is only possible from solar surplus.
        requested_charge = self.max_transfer if a == 1 else 0.0
        charge_amount = min(requested_charge, solar_surplus, float(self.battery_max) - float(battery_old))
        charge_amount = max(0.0, charge_amount)
        charge_failed_penalty = 0.0
        if a == 1 and solar_surplus <= 0.0:
            # Attempted to charge with no solar surplus available.
            charge_failed_penalty = -0.1

        # Discharge is automatically used to cover remaining demand (before grid import).
        # If the agent chose action=2, allow discharge up to max_transfer; otherwise discharge is still allowed
        # but capped by max_transfer to enforce realistic power limits.
        discharge_cap = self.max_transfer
        discharge_amount = min(remaining_demand, discharge_cap, float(battery_old) - float(self.battery_min))
        discharge_amount = max(0.0, discharge_amount)

        # Apply net battery delta with hard limits.
        battery_delta = charge_amount - discharge_amount
        new_battery = float(np.clip(float(battery_old) + battery_delta, self.battery_min, self.battery_max))
        actual_transfer = new_battery - float(battery_old)  # +charge, -discharge

        # 5. Grid import after solar + battery discharge
        grid_import = float(max(0.0, remaining_demand - discharge_amount))

        # 6. Reward (dense): penalize grid import each hour
        reward = -grid_import + charge_failed_penalty

        # 7. Advance time and update next raw state
        self.hour = (self.hour + 1) % self.time_horizon
        next_solar, next_demand = self._get_physics_update(self.hour)
        self.state = np.array([next_solar, next_demand, new_battery, self.hour], dtype=np.float32)

        terminated = False
        truncated = self.hour == 0
        info: dict[str, Any] = {
            "grid_import": grid_import,
            "actual_transfer": actual_transfer,
            "action": a,
            "solar_used_for_demand": solar_used_for_demand,
            "solar_surplus": solar_surplus,
            "battery_charge_amount": charge_amount,
            "battery_discharge_amount": discharge_amount,
            "charge_failed_penalty": charge_failed_penalty,
        }

        def _r2(x: float) -> float:
            return round(float(x), 2)

        # Track this step in history (raw units, rounded)
        self.history.append(
            {
                "hour": _r2(current_hour),
                "solar": _r2(solar),
                "demand": _r2(demand),
                "battery_charge": _r2(new_battery),
                "grid_import": _r2(grid_import),
                "actual_transfer": _r2(actual_transfer),
                "solar_used_for_demand": _r2(solar_used_for_demand),
                "solar_surplus": _r2(solar_surplus),
                "battery_charge_amount": _r2(charge_amount),
                "battery_discharge_amount": _r2(discharge_amount),
                "action": _r2(a),
            }
        )

        if truncated:
            llm_score, llm_feedback = self.critic.generate_score(self.history)
            info["llm_score"] = llm_score
            info["llm_feedback"] = llm_feedback

        return self._get_obs(), float(reward), terminated, truncated, info