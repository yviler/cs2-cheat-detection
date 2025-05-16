# 🗂️ Legacy Code: Initial Version of CS2 Cheat Detection

This folder contains the **first iteration** of the CS2 cheat detection project. It was developed as a basic prototype to test the concept of parsing `.dem` files, extracting aiming data, and identifying suspicious behavior using engineered features.

---

## 🧠 Overview

This early version follows a less modular structure and lacks the scalability and automation of the main version. However, it serves as valuable documentation for the project's evolution.

---

## ⚙️ What It Did

- **Parsed `.dem` files manually** using `demoparser2`
  - Hardcoded paths for specific demo files
  - Extracted kill events and pitch/yaw data
- **Processed one cheater at a time**
  - No batching or folder-wide automation
- **Engineered basic features**
  - Pitch/yaw velocity, acceleration, and jerk
  - Global statistical features (mean, std, min, max)

---

## 🔍 Limitations

- No handling for short segments (< 300 ticks)
- No input normalization or zero-padding
- No class balancing or dynamic labeling
- Lacked folder structure for cheater vs legit separation
- Not modular: scripts combined parsing, processing, and modeling in one place


## 📁 Contents

- `parser.py` – Manual script to parse demo files
- `engineer.py` – First feature engineering prototype
- `model_test.ipynb` – Basic model test on small dataset

> These files are no longer maintained but preserved for reference.

---

## 🔙 See Also

The updated and modular version of the project is available in the main repository root.  
👉 [Back to Main README](../README.md)
