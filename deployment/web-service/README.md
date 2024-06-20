without docker:

gunicorn --bind=0.0.0.0:9696 predict:app

python test.py

with docker:

docker build -t heart-disease-prediction-service:v1 .

docker run -it --rm -p 9696:9696 heart-disease-prediction-service:v1

python test.py