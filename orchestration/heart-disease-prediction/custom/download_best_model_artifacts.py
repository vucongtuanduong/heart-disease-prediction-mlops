if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from mlflow.tracking import MlflowClient
@custom
def transform_custom(*args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your custom logic here

    

    # Initialize the MlflowClient
    client = MlflowClient()

    # Specify the run_id and artifact paths
    run_id = "da93bcc893cd49e09249297415de54e9"
    dict_vectorizer_path = "artifacts/dict_vectorizer.pkl"
    rf_model_path = "artifacts/rf_model.pkl"

    # Specify the local directory to download the artifacts to
    local_dir = "/tmp"

    # Download the artifacts
    client.download_artifacts(run_id, dict_vectorizer_path, local_dir)
    client.download_artifacts(run_id, rf_model_path, local_dir)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
