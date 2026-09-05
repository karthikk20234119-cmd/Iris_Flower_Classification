# Iris Flower Classification

## Overview
This repository contains a complete, submission-ready Machine Learning project for classifying Iris flower species based on physical morphometric measurements. Developed as part of the Data Science Internship assignment (Task 1), this project demonstrates an end-to-end ML workflow incorporating exploratory data analysis (EDA), statistical visualization, feature scaling pipelines, multi-class model training, empirical performance evaluation, and prediction demonstration.

---

## Objective
The objective of this project is to build and evaluate supervised Machine Learning classification models capable of predicting the species of an iris flower:
- **Setosa** (*Iris setosa*)
- **Versicolor** (*Iris versicolor*)
- **Virginica** (*Iris virginica*)

based on four continuous physical measurement features:
1. **Sepal Length** (cm)
2. **Sepal Width** (cm)
3. **Petal Length** (cm)
4. **Petal Width** (cm)

---

## Dataset
- **Source:** Loaded directly from `sklearn.datasets.load_iris()` (No external internet download required).
- **Samples:** 150 instances (50 samples per target class).
- **Features:** 4 numerical continuous variables.
- **Target Classes:** 3 balanced classes (33.33% each).
- **Missing Values:** 0 missing values across all attributes.

---

## Technologies Used
- **Python 3.11+**
- **Jupyter Notebook**
- **pandas** (Data manipulation & DataFrame handling)
- **NumPy** (Numerical arrays & vector operations)
- **matplotlib** & **seaborn** (Statistical visualizations & plots)
- **scikit-learn** (Dataset loading, train-test split, preprocessing pipelines, classification algorithms, and metrics)

---

## Machine Learning Models
Two multi-class classification architectures were trained using Scikit-Learn `Pipeline` objects with `StandardScaler` to prevent data leakage and handle feature scaling:

1. **Logistic Regression:**
   - Preprocessing: `StandardScaler`
   - Model: `LogisticRegression(random_state=42)`
2. **K-Nearest Neighbors (KNN):**
   - Preprocessing: `StandardScaler`
   - Model: `KNeighborsClassifier(n_neighbors=5)`

---

## Evaluation Metrics & Performance Summary

Evaluated on an **80% Training / 20% Testing** stratified split (120 train / 30 test):

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Best)** | **0.9333 (93.33%)** | **0.9333 (93.33%)** | **0.9333 (93.33%)** | **0.9333 (93.33%)** |
| **K-Nearest Neighbors (k=5)** | **0.9333 (93.33%)** | 0.9444 (94.44%) | 0.9333 (93.33%) | 0.9327 (93.27%) |

> **Best Model:** Logistic Regression was selected based on its superior weighted F1-Score (0.9333 vs 0.9327).

---

## Key Findings
1. **Feature Importance:** `petal_length` ($r = 0.95$) and `petal_width` ($r = 0.96$) are the most discriminative features for Iris flower species classification.
2. **Linear Separability:** *Iris setosa* is completely linearly separable from both *Versicolor* and *Virginica* using petal dimensions alone.
3. **Model Convergence:** Scaled feature pipelines achieved 93.33% test accuracy with 100% precision on Setosa and 90%+ precision on Versicolor and Virginica.

---

## Project Structure
```
task1_iris_classification/
│
├── iris_classification.ipynb   # Complete, executed Jupyter Notebook (19 sections)
├── README.md                   # Project documentation & execution guide
├── requirements.txt            # Dependency specification file
├── generate_notebook.py        # Helper script to construct notebook JSON
├── execute_notebook.py         # Helper script for in-place cell execution
├── .venv/                      # Isolated Python virtual environment
└── outputs/                    # Exported high-resolution visualization charts
    ├── boxplots.png
    ├── confusion_matrices.png
    ├── correlation_matrix.png
    ├── feature_distributions.png
    ├── model_accuracy_comparison.png
    ├── pairplot.png
    └── species_count_plot.png
```

---

## How to Run

### 1. Prerequisites
Ensure Python 3.8+ is installed on your system.

### 2. Clone / Open Project Directory
Navigate to the project root directory:
```bash
cd task1_iris_classification
```

### 3. Set Up Virtual Environment & Install Dependencies
Create a virtual environment and install all required libraries from `requirements.txt`:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
Launch the Jupyter Notebook interface in your browser:
```bash
jupyter notebook iris_classification.ipynb
```

### 5. Execute Notebook
- Once Jupyter opens, click **Kernel** -> **Restart & Run All** to run all cells from top to bottom.
- All cell outputs, data tables, Seaborn visual plots, confusion matrix heatmaps, and predictions will render automatically.
