import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify

with open('dict_vectorizer.pkl', 'rb') as f_in:
    dv = pickle.load(f_in)
with open('rf_model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)
with open('scaler.pkl', 'rb') as f_in:
    scaler = pickle.load(f_in)

def prepare_features(ride):
    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([ride])
    features = scaler.transform(df)
    # Create a DataFrame from the numpy array and assign column names from the original DataFrame
    features_df = pd.DataFrame(features, columns=df.columns)
    # Convert the DataFrame back to a dictionary
    features_dict = features_df.to_dict(orient='records')[0]
    return features_dict


def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    probability = model.predict_proba(X)
    return preds, probability


app = Flask('heart-disease-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    preds , probability = predict(features)
    print(features)
    print(preds, probability)

    result = {
        'result': 'ok' if preds[0] == 0 else 'has heart disease',
        'probability': probability[0][1] if preds[0] == 1 else probability[0][0]
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)