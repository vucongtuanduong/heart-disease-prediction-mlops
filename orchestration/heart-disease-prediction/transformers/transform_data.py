if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

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

def normalization(df):
    scaler = StandardScaler()
    df1 = scaler.fit_transform(df)
    df1 = pd.DataFrame(df1, columns = df.columns)
    return df1, scaler

def dict_vectorizer(X_train, X_test):
    dv = DictVectorizer()
    train_dicts = X_train.to_dict(orient = 'records')
    X_train  = dv.fit_transform(train_dicts)
    test_dicts = X_test.to_dict(orient = 'records')
    X_test = dv.transform(test_dicts)
    return X_train, X_test, dv
@transformer
def transform(df, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    # return df;



    X = df.drop(columns = 'target', axis = 1)
    y = df['target']
    
    X,scaler = normalization(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 20, random_state = 42)
    X_train, X_test, dv = dict_vectorizer(X_train, X_test)
    return [X_train, X_test, y_train, y_test, dv, scaler]
    


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'