
# Web Service Deployment

This README provides instructions for deploying the Heart Disease Prediction Service as a web service, both with and without Docker. The service uses Gunicorn as the HTTP server for WSGI applications.

## Overview

The Heart Disease Prediction Service is designed to predict the likelihood of heart disease based on various input features. This service can be deployed in two ways:
- Directly on your local machine using Gunicorn.
- Inside a Docker container for isolation and ease of deployment.

## Folder structure
```bash
├── Dockerfile
├── Pipfile
├── Pipfile.lock
├── README.md
├── dict_vectorizer.pkl
├── predict.py
├── requirements.txt
├── rf_model.pkl
├── scaler.pkl
└── test.py
```
## Deployment Instructions

### Without Docker

1. **Start the Service with Gunicorn**
   
   Install packages

   ```bash
   pip install -r requirements.txt
   ```

   Navigate to the project directory and run the following command to start the service:

   ```bash
   gunicorn --bind=0.0.0.0:9696 predict:app
   ```

   This command starts the Gunicorn server on port 9696, binding it to all network interfaces.

2. **Test the Service**

   After starting the service, you can test it by running:

   ```bash
   python test.py
   ```

   Ensure `test.py` contains the necessary code to send requests to your service and validate responses.

### With Docker

1. **Build the Docker Image**

   In the project directory, build the Docker image using:

   ```bash
   docker build -t heart-disease-prediction-service:v1 .
   ```

   This command creates a Docker image named `heart-disease-prediction-service` with the tag `v1`.

2. **Run the Docker Container**

   Start the service inside a Docker container with:

   ```bash
   docker run -it --rm -p 9696:9696 heart-disease-prediction-service:v1
   ```

   This command runs the Docker container and maps port 9696 of the container to port 9696 on the host, allowing you to access the service at `http://localhost:9696`.

3. **Test the Service**

   With the service running inside Docker, test it by executing:

   ```bash
   python test.py
   ```

   As before, ensure `test.py` is properly configured to test the service.

## Conclusion

Following these instructions, you can deploy the Heart Disease Prediction Service either directly on your machine or within a Docker container. Testing the service ensures it is running correctly and ready to handle prediction requests.
