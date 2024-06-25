#!/usr/bin/env bash

docker-compose up -d
sleep 10

export S3_ENDPOINT_URL='http://localhost:4566'
export INPUT_FILE_PATTERN='s3://heart-disease-prediction/in/test{prefix}.csv'
export OUTPUT_FILE_PATTERN='s3://heart-disease-prediction/out/test{prefix}.csv'
pipenv run aws configure set aws_access_key_id test
pipenv run aws configure set aws_secret_access_key test
pipenv run aws configure set default.region us-east-1
pipenv run aws --endpoint-url="${S3_ENDPOINT_URL}" s3 mb s3://heart-disease-prediction

pipenv run python create_data_integration_test.py
pipenv run python integration_test.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

echo "Integration test completed!"

docker-compose down