import model

def test_prepare_features():
    model_service = model.ModelService()
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
    actual_features = model_service.prepare_features(patient)
    expected_features = {
        "age": -2.3877960151711966,
        "sex": 0.5159524169453267,
        "chest_pain_type": 0.8042418101413419,
        "resting_bp_s": 0.8432462708225252,
        "cholesterol": 0.2853927637810903,
        "fasting_blood_sugar": -0.5513413395776455,
        "resting_ecg": 0.4922407909739158,
        "max_heart_rate": 0.8327535243528628,
        "exercise_angina": 1.2142460751418158,
        "oldpeak": -1.113861169302182,
        "st_slope": -1.0445906831953269
    }
    assert actual_features == expected_features
class ModelMock:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        n = len(X)
        return [self.value] * n


def test_predict():
    model_mock = ModelMock(10.0)
    model_service = model.ModelService(model_mock)

    features = [
        -2.3877960151711966,  # age
        0.5159524169453267,  # sex
        0.8042418101413419,  # chest_pain_type
        0.8432462708225252,  # resting_bp_s
        0.2853927637810903,  # cholesterol
        -0.5513413395776455,  # fasting_blood_sugar
        0.4922407909739158,  # resting_ecg
        0.8327535243528628,  # max_heart_rate
        1.2142460751418158,  # exercise_angina
        -1.113861169302182,  # oldpeak
        -1.0445906831953269  # st_slope
    ]

    # Convert the features list to a list of lists
    features_list = [features]

    actual_prediction = model_service.predict(features_list)
    expected_prediction = 1

    assert actual_prediction == expected_prediction