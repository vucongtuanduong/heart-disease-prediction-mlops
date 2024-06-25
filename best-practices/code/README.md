# Best practices

## Introduction

In this project I apply two basic best practices like unit test and using makefile


## Folder Structure
```bash
.
├── Makefile
├── Pipfile
├── Pipfile.lock
├── README.md
├── __pycache__
│   └── model.cpython-310.pyc
├── batch.py
├── create_data_integration_test.py
├── docker-compose.yml
├── integration_test.py
├── integration_test.sh
├── log.txt
├── model.py
└── tests
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-310.pyc
    │   └── model_test.cpython-310-pytest-8.2.1.pyc
    └── model_test.py
```


## Set up
You just need to run the Makefile to run unit test and integration test

```bash
make setup
```

```bash
make test
```
```bash
make integration_test
```

aws configure

aws --endpoint-url=http://localhost:4566 s3 mb s3://heart-disease-prediction

export S3_ENDPOINT_URL='http://localhost:4566'

export INPUT_FILE_PATTERN="s3://heart-disease-prediction/in/test{preindex}.csv"

export OUTPUT_FILE_PATTERN="s3://heart-disease-prediction/out/test{preindex}.csv"