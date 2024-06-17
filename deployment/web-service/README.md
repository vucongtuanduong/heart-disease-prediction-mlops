without docker: unicorn --bind=0.0.0.0:9696 predict:app

with docker:

docker build -t heart-disease-prediction-service:v1 .

docker run -it --rm -p 9696:9696 heart-disease-prediction-service:v1