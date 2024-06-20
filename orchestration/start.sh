#!/bin/bash

PROJECT_NAME=heart-disease-prediction \
  MAGE_CODE_PATH=/workspaces/heart-disease-prediction-mlops/orchestration/src \
  SMTP_EMAIL=$SMTP_EMAIL \
  SMTP_PASSWORD=$SMTP_PASSWORD \
  docker compose up