if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import pandas as pd 
import requests
from io import BytesIO
from typing import List
@data_loader
# def load_data(*args, **kwargs):
#     """
#     Template code for loading data from any source.

#     Returns:
#         Anything (e.g. data frame, dictionary, array, int, str, etc.)
#     """
#     # Specify your data loading logic here
#     data = pd.read_csv('https://github.com/vucongtuanduong/heart-disease-prediction-mlops/blob/main/data/data.csv')
#     return {data}
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    response = requests.get(
        'https://raw.githubusercontent.com/vucongtuanduong/heart-disease-prediction-mlops/main/data/data.csv'
    )

    if response.status_code != 200:
        raise Exception(response.text)

    df = pd.read_csv(BytesIO(response.content))
    dfs.append(df)
    # print(len(df[df['cholesterol'] == 0]))
    return pd.concat(dfs)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'