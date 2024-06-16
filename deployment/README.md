docker build -t deployment-heart-disease:v1 .

docker run deployment-heart-disease:v1

docker ps -a

docker cp {id}:/app/df_predict_output.csv ./df_predict_output.csv