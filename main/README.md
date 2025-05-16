# CS2 Cheat Detection Pipeline

This repository implements a real-time cheat-detection system for CS2 using LSTM networks on parsed pitch/yaw data. It includes data acquisition, feature engineering, normalization, model training, and evaluation scripts.

## Project Structure

Detailed folder layout is in the repo root. Key directories:

- `data/raw/` — Original `.dem` files and external datasets.
- `data/interim/parsed_csv/` — Parsed tick-by-tick CSVs (before feature engineering).
- `data/features/` — CSVs with computed velocity, acceleration, jerk, and segment stats.
- `data/processed/` — Normalized and split datasets ready for model training.
- `scripts/` — Pipeline scripts, to be run in sequence:
  1. `01_parsing.ipynb`
  2. `02_process_and_engineering.ipynb`
  3. `03_visualize.ipynb`
  4. `04_model.ipynb`
- `notebooks/` — Jupyter notebooks for EDA and visualization.
- `models/` — Saved model weights and architectures.
- `results/` — Training logs, metrics, and plots.
