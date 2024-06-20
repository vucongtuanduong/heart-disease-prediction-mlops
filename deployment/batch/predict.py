import pickle
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler

with open('dict_vectorizer.pkl', 'rb') as f_in:
    dv = pickle.load(f_in)
with open('rf_model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)
with open('scaler.pkl', 'rb') as f_in:
    scaler = pickle.load(f_in)

def load_clean_data(path_name) :
    data = pd.read_csv(path_name)
    data = data.drop_duplicates()
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    return data

def normalization(df):
    df1 = scaler.transform(df)
    df1 = pd.DataFrame(df1, columns = df.columns)
    return df1

def read_data(filename):
    df = load_clean_data(filename)
    
    df1 = normalization(df)
    
    return df, df1

output_file = 'df_predict_output.csv'
df, df1 = read_data('https://raw.githubusercontent.com/vucongtuanduong/heart-disease-prediction-mlops/deployment-test-branch/data/test.csv')
dicts = df1.to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


print(y_pred.mean())

df_result = df.copy()
df_result['Result'] = y_pred
df_result.to_csv(output_file, index=False)