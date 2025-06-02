# CS2 Cheat Detection using Neural Networks

This project is a proof-of-concept system to detect aimbot-like behavior in CS2 (Counter-Strike 2) matches using machine learning—specifically, an LSTM-based neural network trained on engineered features from in-game demo files.

---

## Overview

### Goal  
To build a neural network that can differentiate between cheater and non-cheater gameplay by analyzing player input data (pitch, yaw, velocity, acceleration, etc.) extracted from `.dem` files.

### Pipeline

1. **Parsing**  
   - `.dem` files are parsed into CSV using `demoparser2`.  
   - For every kill event, a 300-tick window is extracted for the attacker.

2. **Processing**  
   - Irrelevant columns like player name are dropped.  
   - Files are organized by class (cheater vs. legit).

3. **Feature Engineering**  
   - First, second, and third derivatives of aim angles (velocity, acceleration, jerk).  
   - Cumulative displacement, statistical summaries, etc.

4. **Modeling**  
   - A two-layer LSTM reads each 300×20 feature segment.  
   - Outputs a probability of the segment being a cheater.

---

## Results (First Test)

| Metric        | Value                  |
|---------------|------------------------|
| Accuracy      | 81%                    |
| Legit Recall  | 91%                    |
| Cheat Recall  | 60%                    |
| F1 Score      | 0.87 (legit), 0.67 (cheat) |

> These results are based on an initial dataset of 104 labeled segments. Performance is expected to improve significantly as more cheater sessions are collected.

---

## Project Structure

The repository implements a real-time cheat-detection pipeline for CS2 using LSTM networks on parsed pitch/yaw data. The main folders and files include:

- `data` — Data root directory
   - `/raw/` — Original `.dem` files and external datasets.  
   - `/interim/parsed_csv/` — Parsed tick-by-tick CSVs (before feature engineering).  
   - `/processed/features/` — Feature-engineered datasets.  
- `scripts/` — Pipeline scripts to run in sequence:  
  1. `00_listSteamid.py`
  2. `01_parser.py`  
  3. `02_process_and_engineer.py`  
  4. `03_model.py`  
- `notebooks/` — Jupyter notebooks for exploratory data analysis and visualization.  
- `legacy_ver/` — The first version of the project with a simpler flow and parser, kept for documentation and evolution tracking.

---

## Next Steps

- Expand cheater dataset across more sessions and cheat types.  
- Tune class weights and decision threshold.  
- Explore attention mechanisms and bidirectional LSTMs.  
- Improve false-positive control to avoid flagging legit players.

---

## Requirements

- Python 3.8+  
- Jupyter  
- demoparser2  
- TensorFlow  
- pandas, numpy, tqdm, matplotlib  

## Installation

Follow the steps below to set up this project locally:

### 1. Clone the repository

```bash
git clone https://github.com/yviler/cs2-cheat-detection.git
cd cs2-cheat-detection
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```
or on Windows
``` bash
venv\Scripts\activate
```

### 3. Install the dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the Project
### 1. (Optional) Insert data into `data/raw/`
### 2. Execute the data scripts pipeline
```bash
python scripts/01_parser.py
python scripts/02_process_and_engineer.py
```
### 3. Execute the Model script
```bash
python scripts/03_model.py
```

