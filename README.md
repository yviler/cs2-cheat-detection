# CS2 Cheat Detection using Neural Networks 🎯

This project is a proof-of-concept system to detect aimbot-like behavior in CS2 (Counter-Strike 2) matches using machine learning—specifically, an LSTM-based neural network trained on engineered features from in-game demo files.

---

## 🧠 Overview

### 🚀 Goal
To build a neural network that can differentiate between cheater and non-cheater gameplay by analyzing player input data (pitch, yaw, velocity, acceleration, etc.) from `.dem` files.

### 🔄 Pipeline

1. **Parsing**
   - `.dem` files are parsed into CSV using `demoparser2`
   - For every kill event, a 300-tick window is extracted for the attacker

2. **Processing**
   - Irrelevant columns like player name are dropped
   - Files are organized by class (cheater vs. legit)

3. **Feature Engineering**
   - First, second, and third derivatives of aim angles (velocity, acceleration, jerk)
   - Cumulative displacement, statistical summaries, etc.

4. **Modeling**
   - A two-layer LSTM reads each 300×20 feature segment
   - Outputs a probability of the segment being a cheater

---

## 📊 Results (First Test)

| Metric        | Value                  |
|---------------|------------------------|
| Accuracy      | 81%                    |
| Legit Recall  | 91%                    |
| Cheat Recall  | 60%                    |
| F1 Score      | 0.87 (legit), 0.67 (cheat) |

> These results are based on an initial dataset of 104 labeled segments. Performance is expected to improve significantly as more cheater sessions are collected.

---

## 🔜 Next Steps

- Expand cheater dataset across more sessions and cheat types
- Tune class weights and decision threshold
- Explore attention mechanisms and bidirectional LSTMs
- Improve false-positive control to avoid flagging legit players

---

## 🧾 Old Version

The first version of the project (in `old_version/`) featured a more basic parser, simpler processing, and a single-script flow. It is preserved for documentation and evolution tracking.  
See [`old_version/README.md`](old_version/README.md) for more details.

---

## 📦 Requirements

- Python 3.8+
- Jupyter
- `demoparser2`
- TensorFlow or PyTorch
- pandas, numpy, tqdm, matplotlib

### 🔧 Installation

Install dependencies with:

```bash
pip install -r requirements.txt
