from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Import functions from your scripts (these need to be on the Python path in Airflow)
# We'll need to slightly refactor train.py to expose the data download function
# For now, let's assume train.py can be run directly.

# Helper function to install pip dependencies within the Airflow task if needed
# Better approach is to bake them into the Docker image or use k8s_pod_operator for isolated envs.
# For local dev, we will ensure they are installed in the common requirements.txt
def install_requirements():
    """Installs required Python packages for the DAG."""
    try:
        import subprocess
        import sys
        # Ensure that the necessary packages are available in the Airflow environment
        # This will install them into the Airflow worker/scheduler/webserver containers.
        # This is generally NOT recommended for production, but convenient for local Docker Compose setup.
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "/opt/airflow/requirements.txt"])
        print("Requirements installed successfully.")
    except Exception as e:
        print(f"Error installing requirements: {e}")
        raise

with DAG(
    dag_id="movie_engagement_ml_pipeline",
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    catchup=False,
    schedule=None, # Define your schedule here (e.g., '@daily')
    tags=["mlops", "movie_recommendation"],
    doc_md="""
    ### Movie Engagement ML Pipeline
    An Airflow DAG to orchestrate the MovieLens 100K dataset download, preprocessing,
    and model training (Logistic Regression & RandomForestClassifier) using MLflow for tracking.
    """
) as dag:
    # Task 1: Install Python dependencies (runs once at the start of a new environment)
    # In a real scenario, these would be baked into a custom Docker image for Airflow.
    install_deps = PythonOperator(
        task_id="install_python_dependencies",
        python_callable=install_requirements,
    )

    # Task 2: Run the training script.
    # We will pass the MLflow tracking URI explicitly to ensure it logs to the shared volume.
    train_model = BashOperator(
        task_id="train_movie_engagement_model",
        bash_command='python /opt/airflow/scripts/train.py', # Full path inside container
        # Ensure that the MLFLOW_TRACKING_URI environment variable is correctly set for this task
        # This should already be picked up from the .env and docker-compose.yaml
        env={
            'MLFLOW_TRACKING_URI': 'file:///opt/airflow/mlruns',
            **os.environ # Pass existing environment variables
        }
    )

    # Define task dependencies
    install_deps >> train_model