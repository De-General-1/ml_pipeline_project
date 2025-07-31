# Use the official Airflow image as a base
FROM apache/airflow:2.8.1-python3.10

# Set the Airflow home directory
ENV AIRFLOW_HOME=/opt/airflow

# Copy your requirements.txt into the container
COPY requirements.txt .

# Install your project-specific Python dependencies
# This ensures that pandas, scikit-learn, mlflow, etc. are available
# in all Airflow components (scheduler, webserver, worker)
RUN pip install --no-cache-dir -r requirements.txt

# You can add other configurations or copy other files if needed
# For example, if you had a custom airflow.cfg:
# COPY airflow.cfg ${AIRFLOW_HOME}/airflow.cfg