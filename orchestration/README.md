# Introduction

This directory contains scripts and configurations for orchestrating the machine learning workflow of the Heart Disease Prediction project. It includes automation for data preprocessing, model training, evaluation, and deployment tasks.

## Overview

Orchestration in machine learning projects is crucial for automating repetitive tasks, ensuring consistency, and facilitating continuous integration and deployment (CI/CD) pipelines. This directory focuses on leveraging workflow orchestration tools to streamline the execution of various components of the project.

## Folder Structure
```bash
├── Dockerfile
├── README.md
├── dict_vectorizer.pkl
├── docker-compose.yml
├── heart-disease-prediction
│   ├── charts
│   │   ├── __init__.py
│   │   ├── feature_profiles_for_ingest.py
│   │   ├── ingest_line_chart_a0.py
│   │   ├── ingest_line_chart_n3.py
│   │   ├── ingest_pie_chart_c7.py
│   │   ├── ingest_pie_chart_d4.py
│   │   ├── ingest_pie_chart_f0.py
│   │   ├── ingest_pie_chart_q0.py
│   │   ├── ingest_pie_chart_t9.py
│   │   ├── ingest_pie_chart_x3.py
│   │   ├── most_frequent_values_for_ingest.py
│   │   └── unique_values_for_ingest.py
│   ├── custom
│   │   ├── __init__.py
│   │   └── download_best_model_artifacts.py
│   ├── data_exporters
│   │   ├── __init__.py
│   │   ├── hyperparameter_tuning.py
│   │   ├── mlflow_register_model.py
│   │   └── train.py
│   ├── data_loaders
│   │   ├── __init__.py
│   │   └── ingest.py
│   ├── metadata.yaml
│   ├── pipelines
│   │   └── data_preparation
│   │       ├── __init__.py
│   │       └── metadata.yaml
│   ├── requirements.txt
│   └── transformers
│       ├── __init__.py
│       └── transform_data.py
├── mlflow
│   └── mlflow.db
├── mlflow.dockerfile
├── rf_model.pkl
├── scaler.pkl
└── start.sh
```

## Contents

- `heart-disease-prediction`: This directory contains all the files during Mage running
- `mlflow`: This directory store mlflow database as sqlite
- `mlflow.dockerfile`: A Dockerfile to run MLFlow using Docker
- `docker-compose.yml`: A docker-compose file to run both MLFlow and Mage

## Getting Started

To run Mage and Docker, please do the following steps:

1. Run `start.sh` to run both MLFlow and Mage using Docker

```bash
./start.sh
```

2. Change to `Ports` and click `Open in Browser` MLFlow at Port 5000 and Mage at Port 6789 

![](../images/orchestration1.png)