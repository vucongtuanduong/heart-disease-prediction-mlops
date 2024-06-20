FROM python:3.10.13-slim

# do stuff here
RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy
COPY ["dict_vectorizer.pkl", "rf_model.pkl", "scaler.pkl", "predict.py", "./"]
CMD ["python", "predict.py"]