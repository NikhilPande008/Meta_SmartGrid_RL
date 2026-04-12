---
title: Meta SmartGrid RL
emoji: 🚀
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
---

# Meta SmartGrid RL: AI-Audited Energy Management

## The Concept
This project is a Reinforcement Learning environment for residential energy optimization. It simulates a household with solar production, demand, and a **battery storage system (50kWh max capacity)**. The agent’s objective is to reduce **grid import** by charging during solar peaks and discharging during the evening demand spike.

## Key Feature: The LLM Critic
Unlike standard environments, we integrate **Llama 3.3 70B (via OpenRouter)** as a “Smart Critic.” It provides:

- **Numerical scoring (1–10)**: sustainability + strategy
- **Natural language audits**: a concise explanation of what the agent did well/poorly

## What’s inside

- `src/meta_smartgrid_rl/env.py`: `SustainableGridEnv` (Gymnasium environment)
- `src/meta_smartgrid_rl/llm_scorer.py`: `GridCritic` (OpenRouter via OpenAI client)
- `test_env.py`: 24-step smoke run (prints totals + `llm_score` / `llm_feedback`)
- `test_logic.py`: `unittest` logic tests (uses a dummy critic; no network)
- `plot_results.py`: plots Solar vs Demand vs Battery Level
- `Dockerfile`: container entrypoint runs `test_env.py`

## How to run (local)

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 test_env.py
```

Run tests:

```bash
python3 test_logic.py
```

Generate the 24-hour plot (saved as `results_plot.png`):

```bash
python3 plot_results.py
```

## Add your API key (OpenRouter)

Create a `.env` file in the repo root:

```bash
OPENROUTER_API_KEY=your_key
```

The critic targets:

- **Base URL**: `https://openrouter.ai/api/v1`
- **Model**: `meta-llama/llama-3.3-70b-instruct`

To avoid spending credits during debugging/CI, you can skip the LLM call:

```bash
SKIP_LLM=1 python3 test_env.py
```

## Docker

Build & run:

```bash
docker build -t smartgrid-rl .
docker run --rm --env-file .env smartgrid-rl
```

## Expert Mode (Judges)

The qualitative critic uses **Llama 3.3 70B** via OpenRouter to produce an end-of-episode score + feedback.

If you are running the Streamlit dashboard in a container, use the `-e` flag to provide your key:

```bash
docker run -p 8501:8501 -e OPENROUTER_API_KEY='your_key_here' meta-smartgrid-app
```