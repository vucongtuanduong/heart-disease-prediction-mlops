if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
import pickle
import mlflow
from mlflow import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.metrics import f1_score, roc_auc_score
@data_exporter
def export_data(X_train, X_test, y_train, y_test, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("heart-disease-experiment")
    with mlflow.start_run():
        mlflow.set_tag("developer", "duongvct");
        
