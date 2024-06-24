
# Model Monitoring for Heart Disease Prediction Service

This README outlines the approach and setup for monitoring the Heart Disease Prediction model in production, specifically utilizing Evidently AI for generating comprehensive monitoring reports.

## Overview

Monitoring is crucial for maintaining the performance, reliability, and accuracy of the Heart Disease Prediction model over time. This document focuses on using Evidently AI to track model performance, data drift, and other key metrics.

## Prerequisites

- Python environment with Evidently AI, Pandas, and Joblib installed.
- Access to the production environment where the model is deployed.
- Historical and current data for generating reports.

## Monitoring Setup with Evidently AI

Evidently AI provides a suite of tools for monitoring data drift, model performance, and more. The setup involves generating reports that can be visualized in Jupyter Notebooks or integrated into web applications.

### Key Metrics to Monitor

- **Data Drift**: Using `DatasetDriftMetric` to detect changes in data distribution over time.
- **Model Performance**: Tracking accuracy, precision, recall, and F1 score through custom metrics.
- **Missing Values**: Utilizing `DatasetMissingValuesMetric` to monitor missing values in the dataset.
- **Feature Correlations**: `ColumnCorrelationsMetric` helps in understanding the relationships between different features.
- **Feature Distributions**: `ColumnQuantileMetric` and `ColumnValuePlot` for analyzing the distributions of input features.

### Steps for Generating Reports

1. **Data Preparation**: Ensure that you have both historical (training) data and recent (production) data available as Pandas DataFrames.
2. **Column Mapping**: Define a `ColumnMapping` object to specify numerical features, categorical features, target column, etc.
3. **Report Generation**: Use the `Report` class to generate reports. You can include various metrics like `ColumnDriftMetric`, `DatasetDriftMetric`, etc., based on your monitoring needs.
4. **Visualization**: Visualize the reports directly in Jupyter Notebooks to analyze the metrics.

### Example Code Snippet

```python
from evidently import ColumnMapping
from evidently.report import Report
import evidently.metrics as ev_metrics

# Load your dataframes (df_train for historical data, df_recent for production data)
# Define your column mapping
column_mapping = ColumnMapping()

# Generate the report
report = Report(metrics=[ev_metrics.ColumnDriftMetric(), ev_metrics.DatasetDriftMetric()])
report.run(reference_data=df_train, current_data=df_recent, column_mapping=column_mapping)

# Visualize the report in Jupyter Notebook
report
```
## Setup instructions
You just have to install packages in requirements.txt
```bash
pip install -r requirements.txt
```

To run Grafana and Adminer, we can use docker-compose

```bash
docker-compose up --build
```
## Conclusion

By leveraging Evidently AI's powerful metrics and reporting capabilities, teams can effectively monitor the Heart Disease Prediction model to ensure its continued accuracy and reliability in production. Regular monitoring and analysis of the reports will help in identifying and addressing issues promptly.
