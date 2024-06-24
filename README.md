# Heart disease prediction project 
This project aims to develop a Machine Learning model capable of predicting the likelihood of a patient having a heart attack based on various health indicators. Utilizing a comprehensive dataset from Kaggle, this model seeks to assist healthcare professionals in identifying at-risk individuals more efficiently.

## Table of Contents

- [Heart disease prediction project](#heart-disease-prediction-project)
- [Table of contents](#table-of-contents)
- [Introduction](#introduction)
- [Tech stack](#tech-stack)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Setup](#setup)
- [Notebook](notebooks/README.md)
- [Orchestration](orchestration/README.md)
- [Batch Deployment](deployment/batch/README.md)
- [Web service Deployment](deployment/web-service/README.md)
- [Monitoring](monitoring/README.md)
- [Best Practices](best-practices/code/README.md)


## Introduction
Heart disease remains one of the leading causes of death globally. Early detection and preventive measures can significantly reduce the risk. This project leverages machine learning techniques to predict heart disease presence in patients, contributing to early diagnosis and better healthcare outcomes.

## Tech stack

- This project I use Python and some helpful libraries like pandas, numpy, matplotlib, seaborn to EDA the dataset and sklearn, xgboost, catboost for data preprocessing, model selection

- Managed the machine learning lifecycle, including experiment tracking, reproducibility, and model deployment, using MLflow 

- Orchestrated the workflow, including data ingestion, transformation, model training, logging to MLFlow , registering model and exporting models, using Mage. I also run both Mage and MLFLow using Dockerfile

- Containerized the application with Docker, ensuring consistency across environments

- Deployed the model for batch predictions using Docker and for real-time predictions using Gunicorn, Flask, and Docker.

- Used Evidently for model monitoring, creating reports to ensure model performance.

- Applied best practices such as unit testing and utilizing Makefile for build automation.


## Dataset
I use this Kaggle [dataset](https://www.kaggle.com/datasets/mexwell/heart-disease-dataset/data) 

**Heart Disease Dataset Attribute Description** 



|**S.No.** |**Attribute** |**Code given** |**Unit** |**Data type** |
| - | - | - | - | - |
|1 |age |Age |in years |Numeric |
|2 |sex |Sex |1, 0  |Binary |
|3 |chest pain type |chest pain type |1,2,3,4 |Nominal |
|4 |resting blood pressure |resting bp s |in mm Hg |Numeric |
|5 |serum cholesterol |cholesterol |in mg/dl |Numeric |
|6 |fasting blood sugar |fasting blood sugar |1,0 > 120 mg/dl |Binary |
|7 |resting electrocardiogram results |resting ecg |0,1,2 |Nominal |
|8 |maximum heart rate achieved |max heart rate |71–202 |Numeric |
|9 |exercise induced angina |exercise angina |0,1 |Binary |
|10 |oldpeak =ST |oldpeak |depression |Numeric |
|11 |the slope of the peak exercise ST segment |ST slope |0,1,2 |Nominal |
|12 |class |target |0,1 |Binary |

**Description of Nominal Attributes** 



|**Attribute** |**Description** |
| - | - |
|Sex |1 = male, 0= female; |
|Chest Pain Type |<p>-- Value 1: typical angina </p><p>-- Value 2: atypical angina </p><p>-- Value 3: non-anginal pain </p><p>-- Value 4: asymptomatic </p>|
|Fasting Blood sugar |(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) |
|Resting electrocardiogram results |<p>-- Value 0: normal </p><p>-- Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) </p><p>-- Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria </p>|
|Exercise induced angina|1 = yes; 0 = no |
|the slope of the peak exercise ST segment|<p>-- Value 1: upsloping </p><p>-- Value 2: flat </p><p>-- Value 3: downsloping </p>|
|class |1 = heart disease, 0 = Normal |


## Project Structure
```bash
.
├── README.md
├── best-practices
│   └── code
│       ├── Makefile
│       ├── Pipfile
│       ├── Pipfile.lock
│       ├── README.md
│       ├── model.py
│       └── tests
│           ├── __init__.py
│           └── model_test.py
├── data # Store data used for the project
│   ├── create_random_test_data.py
│   ├── data.csv
│   └── test.csv
├── deployment # deploy the model using batch and web service (flask and gunicorn)
│   ├── batch
│   │   ├── Dockerfile
│   │   ├── Pipfile
│   │   ├── Pipfile.lock
│   │   ├── README.md
│   │   ├── df_predict_output.csv
│   │   ├── dict_vectorizer.pkl
│   │   ├── predict.py
│   │   ├── rf_model.pkl
│   │   └── scaler.pkl
│   └── web-service
│       ├── Dockerfile
│       ├── Pipfile
│       ├── Pipfile.lock
│       ├── README.md
│       ├── dict_vectorizer.pkl
│       ├── predict.py
│       ├── requirements.txt
│       ├── rf_model.pkl
│       ├── scaler.pkl
│       └── test.py
├── images
│   ├── batch-deployment.png
│   └── orchestration1.png
├── model # store model exported from MLFlow
│   ├── dict_vectorizer.pkl
│   ├── rf_model.pkl
│   └── scaler.pkl
├── monitoring # monitor the ML Model
│   ├── README.md
│   ├── docker-compose.yml
│   ├── heart-disease-predict-monitor.ipynb
│   ├── requirements.txt
│   └── workspace
│       └── 0d494e17-0f60-4176-8376-52857a34bcc9
│           ├── metadata.json
│           └── snapshots
│               ├── e061cc52-43b1-4235-bc1b-1d2a577cd967.json
│               └── ecc1fa28-9b34-4673-83b8-574432245547.json
├── notebooks # notebooks for EDA and data preprocessing
│   ├── README.md
│   ├── catboost_info
│   │   ├── catboost_training.json
│   │   ├── learn
│   │   │   └── events.out.tfevents
│   │   ├── learn_error.tsv
│   │   └── time_left.tsv
│   ├── model.ipynb
│   └── requirements.txt
└── orchestration # mage and mlflow in the same docker container
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
    ├── mage_data
    ├── mlflow
    │   └── mlflow.db
    ├── mlflow.dockerfile
    ├── rf_model.pkl
    ├── scaler.pkl
    ├── scripts
    │   └── database
    ├── src
    │   └── mage_data
    └── start.sh


```

## Setup

Detailed setup to reproduce results will be provided in each directory but the first thing you can do is using Github Codespaces, creating codespaces

Then create conda environment:

```bash
conda create -n test-env python==3.10.13
```

```bash
conda init 
```

```bash
conda activate test-env
```