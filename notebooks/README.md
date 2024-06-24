# Notebooks Directory

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

The notebooks in this directory serve multiple purposes:

- **Data Exploration**: Understand the dataset by visualizing the data distribution, identifying patterns, and detecting outliers.
- **Feature Engineering**: Create and select features that improve model performance.
- **Model Prototyping**: Experiment with different machine learning models to predict heart disease.
- **Evaluation**: Assess model performance using appropriate metrics.

## Notebooks Description

- `EDA.ipynb`: Exploratory Data Analysis - A notebook dedicated to understanding the data through visualization and statistics.
- `Feature_Engineering.ipynb`: Feature Engineering - Techniques applied to the data to improve model input.
- `Model_Prototyping.ipynb`: Model Prototyping - Experimentation with various machine learning models.
- `Model_Evaluation.ipynb`: Model Evaluation - Evaluation of model performance and metrics.

## Getting Started

To get started with these notebooks:

1. Ensure you have Jupyter installed in your environment. If not, you can install it using pip:

```bash
pip install jupyter