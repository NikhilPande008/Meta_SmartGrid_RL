from __future__ import annotations

import os
from typing import Any, Optional

# Load .env as early as possible so OPENROUTER_API_KEY is available at import time.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass


class GridCritic:
    def __init__(
        self,
        *,
        model: str = "meta-llama/llama-3.3-70b-instruct",
        battery_capacity_kwh: float = 50.0,
        max_solar_potential_kwh: float = 12.0,
    ) -> None:
        """
        LLM-based critic (OpenRouter via OpenAI client).
        """

        self.model = model
        self.battery_capacity_kwh = float(battery_capacity_kwh)
        self.max_solar_potential_kwh = float(max_solar_potential_kwh)

        self.api_key = os.getenv("OPENROUTER_API_KEY")

        self.client: Optional[Any] = None
        self._client_init_error: Optional[str] = None
        try:
            from openai import OpenAI  # type: ignore

            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                max_retries=2,
                timeout=30.0,
            )
        except Exception as e:
            self.client = None
            self._client_init_error = str(e)

    def _build_log_summary(self, history: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for h in history:
            try:
                lines.append(
                    "Hour {hour}: Solar={solar:.2f}kWh, Demand={demand:.2f}kWh, "
                    "Battery={battery:.2f}kWh, GridImport={grid:.2f}kWh".format(
                        hour=int(float(h.get("hour", 0))),
                        solar=float(h.get("solar", 0.0)),
                        demand=float(h.get("demand", 0.0)),
                        battery=float(h.get("battery_charge", 0.0)),
                        grid=float(h.get("grid_import", 0.0)),
                    )
                )
            except Exception:
                # Keep going even if a row is malformed
                continue
        return "\n".join(lines)

    def _build_summary(self, history: list[dict[str, Any]]) -> str:
        total_solar = sum(float(h.get("solar", 0.0)) for h in history)
        total_demand = sum(float(h.get("demand", 0.0)) for h in history)
        total_grid_import = sum(float(h.get("grid_import", 0.0)) for h in history)
        total_grid_saved = max(0.0, total_demand - total_grid_import)
        max_solar_potential = max([float(h.get("solar", 0.0)) for h in history] + [0.0])

        return (
            "## Summary (precomputed)\n"
            f"- Total Solar Produced: {total_solar:.2f} kWh\n"
            f"- Total Demand: {total_demand:.2f} kWh\n"
            f"- Total Grid Import: {total_grid_import:.2f} kWh\n"
            f"- Total Grid Saved (Demand - Grid Import): {total_grid_saved:.2f} kWh\n"
            f"- Max Solar Potential (peak observed): {max_solar_potential:.2f} kWh\n"
        )

    def generate_score(self, history: list[dict[str, Any]]) -> tuple[int, str]:
        if os.getenv("SKIP_LLM", "").strip().lower() in {"1", "true", "yes", "y"}:
            return 0, "LLM skipped (SKIP_LLM=1) to save credits."

        if not self.api_key:
            return 0, "Missing API Key in .env file"

        if self.client is None:
            return 0, (self._client_init_error or "OpenAI client not initialized")

        log_summary = self._build_log_summary(history)
        summary = self._build_summary(history)

        try:
            # Force connection attempt: this must try to call the OpenRouter endpoint.
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an Energy Audit Expert.\n"
                            f"You are evaluating a residential energy controller with a {self.battery_capacity_kwh:.0f}kWh Residential Battery.\n"
                            "All time steps are 1 hour and all values are in kWh.\n"
                            "System Stats:\n"
                            f"- Max Battery Capacity: {self.battery_capacity_kwh:.0f}kWh\n"
                            f"- Max Solar Potential (theoretical): {self.max_solar_potential_kwh:.2f} kWh\n"
                            "- Max Solar Potential (observed): see Summary\n"
                            "A good agent charges during solar peaks and saves that energy for the 6 PM - 10 PM demand spike.\n"
                            "Review the 24-hour grid log below and rate the agent from 1-10 on:\n"
                            "- SUSTAINABILITY (minimizing grid import / dirty energy)\n"
                            "- STRATEGY (charging/discharging at the right times)\n"
                            "Important action note: Action 1 (CHARGE) is only useful if solar > 0 and preferably when solar surplus exists.\n"
                            "Provide a 1-sentence reasoning.\n"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{summary}\n"
                            "## 24h Log\n"
                            f"{log_summary}\n\n"
                            "Format: Score: X | Feedback: Y"
                        ),
                    },
                ],
            )

            text = response.choices[0].message.content or ""
            score_part = text.split("|")[0]
            digits = "".join(ch for ch in score_part if ch.isdigit())
            score = int(digits) if digits else 0
            return score, text
        except Exception as e:
            # Return the raw error message so issues like "402 Payment Required"
            # are not obscured.
            return 0, str(e)

