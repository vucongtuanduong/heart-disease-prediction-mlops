import os
import json
import base64
# import mlflow
import pickle
import pandas as pd
import sys
from pathlib import Path
class ModelService:
    def __init__(self, model = None):
        base_path = Path(__file__).parent.parent.parent  # get the parent directory of the current file
        file_path1 = base_path / "model/dict_vectorizer.pkl"  # construct the full path
        file_path2 = base_path / "model/rf_model.pkl"  # construct the full path
        file_path3 = base_path / "model/scaler.pkl"  # construct the full path
        with open(file_path1, 'rb') as f_in:
            dv = pickle.load(f_in)
        if (model != None):
            with open(file_path2, 'rb') as f_in:
                model = pickle.load(f_in)
        with open(file_path3, 'rb') as f_in:
            scaler = pickle.load(f_in)
        self.model = model
        self.dv = dv
        self.scaler = scaler

    def prepare_features(self, ride):
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame([ride])
        features = self.scaler.transform(df)
        # Create a DataFrame from the numpy array and assign column names from the original DataFrame
        features_df = pd.DataFrame(features, columns=df.columns)
        # Convert the DataFrame back to a dictionary
        features_dict = features_df.to_dict(orient='records')[0]
        return features_dict
    def predict(self, features):
        pred = self.model.predict(features)
        return pred[0]