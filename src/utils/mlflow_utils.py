import mlflow
import mlflow.tensorflow
from datetime import datetime
import os

def setup_mlflow(experiment_name: str = "lung_cancer_detection"):
    """
    Setup MLflow tracking
    Args:
        experiment_name (str): Name of the experiment
    Returns:
        str: Active run ID
    """
    # Set tracking URI to local 'mlruns' directory
    mlflow.set_tracking_uri("file:" + os.path.join(os.getcwd(), "mlruns"))
    
    # Create or get existing experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Error setting up MLflow experiment: {e}")
        return None

    # Start a new run
    run = mlflow.start_run(experiment_id=experiment_id)
    return run.info.run_id

def log_model_parameters(model, **kwargs):
    """
    Log model parameters to MLflow
    Args:
        model: Tensorflow model
        **kwargs: Additional parameters to log
    """
    try:
        # Log model architecture
        mlflow.log_param("model_summary", model.summary())
        
        # Log additional parameters
        for key, value in kwargs.items():
            mlflow.log_param(key, value)
    except Exception as e:
        print(f"Error logging parameters: {e}")

def log_metrics(metrics: dict, step: int = None):
    """
    Log metrics to MLflow
    Args:
        metrics (dict): Dictionary of metrics to log
        step (int): Step number (optional)
    """
    try:
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value, step=step)
    except Exception as e:
        print(f"Error logging metrics: {e}")

def save_model(model, run_id: str = None):
    """
    Save model to MLflow
    Args:
        model: Tensorflow model
        run_id (str): MLflow run ID
    """
    try:
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.tensorflow.log_model(model, "model")
        else:
            mlflow.tensorflow.log_model(model, "model")
    except Exception as e:
        print(f"Error saving model: {e}")

def end_run():
    """End the current MLflow run"""
    try:
        mlflow.end_run()
    except Exception as e:
        print(f"Error ending run: {e}")
