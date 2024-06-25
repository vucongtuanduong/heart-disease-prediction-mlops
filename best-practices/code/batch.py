#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd
import os
import random
import numpy as np
from pathlib import Path
def get_input_path(preindex, prefix = 'default-prefix'):
    default_input_pattern = f'https://raw.githubusercontent.com/vucongtuanduong/heart-disease-prediction-mlops/dev-test/data/test{preindex}.csv'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(preindex = preindex, prefix = preindex)


def get_output_path(preindex, prefix = 'default-prefix'):
    default_output_pattern = f's3://heart-disease-prediction/preindex = {preindex}/predictions.csv'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(preindex = preindex, prefix = preindex)
def read_data(filename):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    if S3_ENDPOINT_URL is not None:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }
        df = pd.read_csv(filename, storage_options=options)
    else:
        df = pd.read_csv(filename)
    return df

def prepare_data(df):
    base_path = Path(__file__).parent.parent.parent  # get the parent directory of the current file
    file_path1 = base_path / "model/dict_vectorizer.pkl"  # construct the full path
    file_path3 = base_path / "model/scaler.pkl"  # construct the full path
    with open(file_path1, 'rb') as f_in:
        dv = pickle.load(f_in)
    with open(file_path3, 'rb') as f_in:
        scaler = pickle.load(f_in)
    df_columns = df.columns
    df = scaler.transform(df)
    df = pd.DataFrame(df, columns = df_columns)
    dicts = df.to_dict(orient='records')    
    df = dv.transform(dicts)
    return df

def save_data(df, output_filename):
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')
    if S3_ENDPOINT_URL is not None:
        options = {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }
        df.to_csv(output_filename, index=False, storage_options = options)
    else:
        df.to_csv(output_filename,index=False)

def main(preindex):

    input_file = get_input_path(preindex)
    output_file = get_output_path(preindex)
    
    base_path = Path(__file__).parent.parent.parent  # get the parent directory of the current file
    file_path2 = base_path / "model/rf_model.pkl"  # construct the full path
    with open(file_path2, 'rb') as f_in:
        model = pickle.load(f_in)
    df = read_data(input_file)
    df1 = prepare_data(df)
    y_pred = model.predict(df1)
    df_result = df.copy()
    df_result['predicted_target'] = y_pred
    save_data(df_result, output_file)


    

if __name__ == "__main__":
    preindex = int(sys.argv[1])
    main(preindex)