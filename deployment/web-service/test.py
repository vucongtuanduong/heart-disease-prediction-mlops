import requests

patient = {
    "age": 31,
    "sex": 1,
    "chest_pain_type": 4,
    "resting_bp_s": 148,
    "cholesterol": 230,
    "fasting_blood_sugar": 0,
    "resting_ecg": 1,
    "max_heart_rate": 158,
    "exercise_angina": 1,
    "oldpeak": -0.3,
    "st_slope": 1
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=patient)

# Check if the response contains JSON data
if response.headers['Content-Type'] == 'application/json':
    print(response.json())
else:
    print('Response is not in JSON format. Response received: ', response.text)