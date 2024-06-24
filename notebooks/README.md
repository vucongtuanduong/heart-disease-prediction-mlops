# Introduction

This directory contains Jupyter notebook for the Heart Disease Prediction project. The notebook is used for exploratory data analysis, model prototyping, and various model building experiments related to predicting heart disease.

## Folder Structure
```bash
├── README.md
├── catboost_info 
│   ├── catboost_training.json
│   ├── learn
│   │   └── events.out.tfevents
│   ├── learn_error.tsv
│   └── time_left.tsv
├── model.ipynb
└── requirements.txt

```
## Overview

The notebook in this directory serve multiple purposes:

- **Data Preprocessing and Exploration**: Understand the dataset by visualizing the data distribution, identifying patterns, and detecting outliers.
- **Data normalization**: Normalize data using StandardScaler and DictVectorizer
- **Model Prototyping**: Experiment with different machine learning models including Logistic Regression, Random Forest, XGBoost, CatBoost to predict heart disease.
- **Evaluation**: Assess model performance using appropriate metrics like F1 Score and ROC AUC Score.


## Getting Started

To get started with these notebooks:

1. Install packages

```bash
pip install -r requirements.txt
```
2. Run Jupyter Notebook 

```bash
jupyter notebook 
```

3. Get the token and enter token to access notebook