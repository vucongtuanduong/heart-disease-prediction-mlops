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
    # check_list = [0.592,0.56,0.598,0.604,0.614,0.596,0.616,0.616,0.584,0.596,0.568]
    for i in range(1, 12, 1):
        input_file = batch.get_input_path(i)
        output_file = batch.get_output_path(i)
        os.system(f'python batch.py {i}')
        df_actual = pd.read_csv(output_file, storage_options=options)
        print(df_actual['predicted_target'].mean())
        # assert abs(df_actual['predicted_target'].mean()- check_list[i - 1]), 'Wrong prediction'
    

if __name__ == "__main__":
    test_create_data()