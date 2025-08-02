# Credit-Card-Fraud-DL-Model
This project presents a deep learning-based approach to detecting fraudulent credit card transactions. It blends **unsupervised anomaly detection** (Autoencoders) with **supervised classification** (Convolutional Neural Networks) to enhance detection accuracy on imbalanced real-world data.

---

## 🔍 Project Highlights

Credit card fraud is rare but highly damaging. This system aims to intelligently identify such anomalies by:

- Using an **Autoencoder** to learn patterns of normal transactions and detect outliers based on reconstruction error.
- Training a **1D CNN** to classify transactions using reshaped, normalized feature data.
- Applying **SMOTE** to balance the highly skewed dataset.
- Combining both models in a **weighted ensemble** for improved performance.

---

## 🗃️ Dataset Overview

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Records**: 284,807 transactions  
- **Fraud Cases**: 492 (~0.17%)  
- **Features**:
  - 28 anonymized PCA-transformed columns (`V1` to `V28`)
  - `Time`, `Amount`, and the target `Class` (0 = Normal, 1 = Fraud)

---

## 📁 Repository Contents

- `Jupyter Source Code.ipynb` – complete training and evaluation workflow:
  - Data cleaning & preprocessing
  - SMOTE balancing
  - Autoencoder model
  - CNN model
  - Final ensemble logic
  - Model saving (`.h5`, `.pkl`)

---

## ⚙️ Installation

Make sure you have Python installed, then install the required libraries:

```bash
pip install pandas numpy scikit-learn imbalanced-learn tensorflow keras joblib matplotlib seaborn
