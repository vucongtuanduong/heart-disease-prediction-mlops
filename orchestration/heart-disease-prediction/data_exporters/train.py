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
def normalization(df):
    scaler = StandardScaler()
    df1 = scaler.fit_transform(df)
    df1 = pd.DataFrame(df1, columns = df.columns)
    return df1

def dict_vectorizer(X_train, X_test):
    dv = DictVectorizer()
    train_dicts = X_train.to_dict(orient = 'records')
    X_train  = dv.fit_transform(train_dicts)
    test_dicts = X_test.to_dict(orient = 'records')
    X_test = dv.transform(test_dicts)
    return X_train, X_test
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
    X = df.drop(columns = 'target', axis = 1)
    y = df['target']
    
    X = normalization(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 20, random_state = 42)
    X_train, X_test = dict_vectorizer(X_train, X_test)
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("heart-disease-experiment")
    # Logistic Regression
    with mlflow.start_run(run_name="Logistic Regression"):
        mlflow.set_tag("developer", "duongvct");
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        mlflow.log_param("train-data-path", "/workspaces/heart-disease-prediction-mlops/data/data.csv")
        mlflow.log_param("f1 score", f1)
        mlflow.log_param("roc auc score", roc_auc)
        mlflow.log_metric("f1 score", f1)
        mlflow.log_metric("roc auc score", roc_auc)

    # Random Forest
    with mlflow.start_run(run_name="Random Forest"):
        mlflow.set_tag("developer", "duongvct");
        mlflow.log_param("train-data-path", "/workspaces/heart-disease-prediction-mlops/data/data.csv")
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        mlflow.log_param("f1 score", f1)
        mlflow.log_param("roc auc score", roc_auc)
        mlflow.log_metric("f1 score", f1)
        mlflow.log_metric("roc auc score", roc_auc)
    # XGBoost
    with mlflow.start_run(run_name="XGBoost"):
        mlflow.set_tag("developer", "duongvct");
        mlflow.log_param("train-data-path", "/workspaces/heart-disease-prediction-mlops/data/data.csv")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Specify the parameters via map
        param = {
            'max_depth': 5,  # the maximum depth of each tree
            'eta': 0.2,  # the training step for each iteration
            'objective': 'multi:softprob',  # error evaluation for multiclass training
            'num_class': 2}  # the number of classes that exist in this dataset
        num_round = 20  # the number of training iterations

        # Train the model
        bst = xgb.train(param, dtrain, num_round)

        # Make prediction
        preds = bst.predict(dtest)
        # Choose the best prediction
        best_preds = np.asarray([np.argmax(line) for line in preds])

        #f1 score
        f1 = f1_score(y_test, best_preds)
        mlflow.log_metric("f1 score", f1)

        # Binarize the output
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test_bin = lb.transform(y_test)
        best_preds_bin = lb.transform(best_preds)

        # Calculate the ROC AUC score
        roc_auc = roc_auc_score(y_test_bin, best_preds_bin, average='macro')
        mlflow.log_metric("roc auc score", roc_auc)
        mlflow.log_param("f1 score", f1)
        mlflow.log_param("roc auc score", roc_auc)

