# K-Means Customer Segmentation

Machine learning homework project for customer segmentation using RFM features and k-Means clustering on the Online Retail II dataset.

## Overview

The project builds customer segments from transaction data using:

- RFM feature engineering: Recency, Frequency, and Monetary value
- Skewness analysis and log transformation
- Standardization before clustering
- Elbow and silhouette analysis for selecting `k`
- PCA-based visualization of customer clusters

## Project Structure

```text
.
├── train.py              # Runs the full training and clustering pipeline
├── eval.py               # Evaluates and summarizes generated segments
├── dataset.py            # Loads data and builds RFM features
├── preprocessing.py      # Handles transformations and scaling
├── clustering.py         # k-Means and cluster selection utilities
├── utils.py              # Plotting and output helpers
├── generate_report.py    # Generates the project report
├── requirements.txt      # Python dependencies
└── results/              # Generated metrics, plots, and segment outputs
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place the Online Retail II dataset at:

```text
dataset/online_retail_II.xlsx
```

Dataset source: https://archive.ics.uci.edu/dataset/502/online+retail+ii

## Usage

Run the full analysis:

```bash
python train.py
```

Evaluate the generated clusters:

```bash
python eval.py
```

## Author

Mohammed Izedin Mohammed
