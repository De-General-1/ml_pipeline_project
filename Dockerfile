FROM apache/airflow:2.8.0-python3.10
USER airflow
RUN pip install --no-cache-dir mlflow boto3 scikit-learn pandas numpy
