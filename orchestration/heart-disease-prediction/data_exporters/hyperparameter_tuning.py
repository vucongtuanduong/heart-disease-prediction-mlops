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
from sklearn.preprocessing import LabelBinarizer
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope

@data_exporter
def export_data(df, *args, **kwargs):
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
    X_train = df[0]
    X_test = df[1]
    y_train = df[2]
    y_test = df[3]
    dv = df[4]
    num_trials = 15
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("heart-disease-experiment")
    RF_PARAMS = ['max_depth', 'n_estimators',
             'min_samples_split', 'min_samples_leaf',
             'random_state']
    
    def objective(params):
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        max_depth = rf.get_params()['max_depth']
        n_estimators = rf.get_params()['n_estimators']
        min_samples_split = rf.get_params()['min_samples_split']
        min_samples_leaf = rf.get_params()['min_samples_leaf']
        random_state = rf.get_params()['random_state']

        with mlflow.start_run(nested=True):
            mlflow.log_param('n_estimators', n_estimators)
            mlflow.log_param('min_samples_split', min_samples_split)
            mlflow.log_param('min_samples_leaf', min_samples_leaf)
            mlflow.log_param('random_state', random_state)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc_score", roc_auc)

        # Return the negative of the metrics as the 'loss' since hyperopt minimizes the objective
        return {'loss': -max(f1, roc_auc), 'f1_score': f1, 'roc_auc_score': roc_auc, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )