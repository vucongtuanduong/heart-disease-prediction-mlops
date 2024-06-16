import pandas as pd
import random
import numpy as np

# Define data
data = {
    'age': [random.randint(20, 80) for _ in range(500)],
    'sex': [random.choice([0, 1]) for _ in range(500)],
    'chest_pain_type': [random.choice([1, 2, 3, 4]) for _ in range(500)],
    'resting_bp_s': [random.randint(80, 200) for _ in range(500)],
    'cholesterol': [random.randint(100, 350) for _ in range(500)],
    'fasting_blood_sugar': [random.choice([0, 1]) for _ in range(500)],
    'resting_ecg': [random.choice([0, 1, 2]) for _ in range(500)],
    'max_heart_rate': [random.randint(71, 202) for _ in range(500)],
    'exercise_angina': [random.choice([0, 1]) for _ in range(500)],
    'oldpeak': [np.round(random.uniform(-2.6, 6.2), 1) for _ in range(500)],  # Values from -2.6 to 6.2 with step 0.1
    'st_slope': [random.choice([1, 2, 3]) for _ in range(500)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Write DataFrame to CSV file
df.to_csv('test.csv', index=False)