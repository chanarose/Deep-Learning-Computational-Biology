# 🧬 Deep Learning for RNA Binding Prediction

This project was created as part of the *Deep Learning in Computational Biology* course at Bar-Ilan University, under the supervision of Prof. Yaron Orenstein.  
It focuses on predicting RNAcompete binding intensities from HTR-SELEX data using neural networks.

> **Authors:** Chana Rosenblum and Nir Koren
> **Grade Received:**  ✅ 95  
> **Date:** August 2024

---

## 🎯 Project Goal

To develop a deep learning model capable of predicting the binding intensity between RNA sequences and RNA-binding proteins (RBPs), by learning from HTR-SELEX data and evaluating against RNAcompete measurements.

---

## 🧪 Datasets

| Source      | Description |
|-------------|-------------|
| **HTR-SELEX** | Training dataset with oligo sequences and cycle labels (1–4) for 38 proteins |
| **RNAcompete** | Testing dataset with ~250K RNA sequences (length 40) and associated binding intensities |

Each RNA sequence is encoded and padded to a fixed length of 50 using a custom tokenizer for A, C, G, U.

---

## 🧠 Model Architecture

A custom neural network combining:

- **Embedding layer** (128-dim)
- **Conv1D layer** (128 filters, kernel size = 3)
- **BatchNorm + ReLU**
- **Adaptive Average Pooling**
- **Fully connected layer (128 → 1)**
- **Optimizer**: AdamW (lr=0.0001, weight decay=0.05)
- **Loss**: MSE  
- **Epochs**: 3  
- **Batch Size**: 256  

---

## 📉 Baselines vs Our Model

To evaluate performance, we compared our results to a simple k-mer frequency-based predictor.  
Our deep model achieved significantly better Pearson correlations and AUPR scores on most RBPs.

| Metric | Baseline Avg | Our Model Avg |
|--------|--------------|----------------|
| **Pearson Corr.** | 0.1745 | 0.2274 |
| **Best RBP AUPR** | 0.5783 | 0.5783 (RBP5) |

Some RBPs (e.g., RBP3, RBP11, RBP24) showed dramatic improvements.

---

## 📉 What Didn't Work

- BERT / SpliceBERT models underperformed
- Attention-based models lacked generalization
- Over-parameterized CNNs led to overfitting
- Tokenizing into k-mers (6-8) yielded better MSE but hurt Pearson scores

---

## ⚙️ Performance

- Avg runtime: 4 minutes per RBP
- Memory: 3GB CPU / ~370MB GPU (NVIDIA P100)
- Frameworks: PyTorch, Python 3.9



