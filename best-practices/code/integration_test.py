import os
import pandas as pd
from datetime import datetime
import batch

def test_create_data():
    S3_ENDPOINT_URL = os.getenv('S3_ENDPOINT_URL')

    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }
    for i in range(1, 12, 1):
        input_file = get_input_path(i)
        output_file = get_output_path(i)
        os.system('python batch_q4_q6.py 2023 1')
        df_actual = pd.read_parquet(output_file, storage_options=options)
        print(df_actual['predicted_duration'].sum())
        assert abs(df_actual['predicted_duration'].sum() - 36.28) < 0.1, 'Wrong prediction'
    

if __name__ == "__main__":
    test_create_data()