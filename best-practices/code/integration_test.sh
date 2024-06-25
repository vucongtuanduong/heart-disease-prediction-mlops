#!/usr/bin/env bash

docker-compose up -d
sleep 10

export S3_ENDPOINT_URL='http://localhost:4566'
export INPUT_FILE_PATTERN='s3://heart-disease-prediction/in/test{prefix}.csv'
export OUTPUT_FILE_PATTERN='s3://heart-disease-prediction/out/test{prefix}.csv'

aws --endpoint-url="${S3_ENDPOINT_URL}" s3 mb s3://heart-disease-prediction

pipenv run python integration_test.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

echo "Integration test completed!"

docker-compose down