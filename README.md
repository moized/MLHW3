# 📈 Retail Customer Segmentation using RFM & K-Means Clustering

## 📌 Overview
This project delivers an end-to-end unsupervised machine learning pipeline to segment retail customers based on their historical transaction patterns. By extracting **RFM (Recency, Frequency, Monetary)** metrics from large-scale e-commerce sheets, the framework cleans data, handles outliers, optimizes cluster thresholds, and projects high-dimensional matrices onto interpretable interfaces using **PCA (Principal Component Analysis)**.

---

## 📁 Project Structure
The repository is engineered into a modular, production-ready machine learning framework following clean software engineering practices:

```text
MLHW3/
├── data/              # Raw transaction sheets (online_retail_II.xlsx)
├── src/               # Core analytical Python modules
│   ├── train.py       # Runs the full training and clustering pipeline
│   ├── eval.py        # Evaluates and summarizes generated segments
│   ├── dataset.py     # Loads data and builds RFM features
│   ├── preprocessing.py # Handles transformations and scaling
│   ├── clustering.py  # k-Means and cluster selection utilities
│   ├── utils.py       # Plotting and output helpers
│   └── generate_report.py # Generates the project report asset
├── reports/           # Business intelligence outputs and data visualization models
│   ├── customer_segmentation_report.html  # Main high-fidelity analytical report
│   └── figures/       # Elbow curves, Silhouette profiles, and PCA cluster maps
└── requirements.txt   # Python dependencies