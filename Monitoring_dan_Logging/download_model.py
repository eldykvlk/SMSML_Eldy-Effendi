import mlflow
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


mlflow_tracking_uri = "http://127.0.0.1:5000" 

mlflow.set_tracking_uri(mlflow_tracking_uri)
logging.info(f"MLflow Tracking URI set to: {mlflow_tracking_uri}")


TARGET_RUN_ID = "706aaef625974b69a7a525a434874b23"


ARTIFACT_PATH_IN_RUN = "walmart_sales_dt_model"


LOCAL_DOWNLOAD_DIR = "mlruns_downloaded"

try:
    # Mengunduh artefak
    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=TARGET_RUN_ID,
        artifact_path=ARTIFACT_PATH_IN_RUN,
        dst_path=LOCAL_DOWNLOAD_DIR
    )
    logging.info(f"Artifacts downloaded successfully to: {downloaded_path}")

    # Path model yang sekarang ada di lokal Anda
    # Ini adalah path yang akan Anda gunakan untuk `mlflow models serve`
    # dan di `prometheus_exporter.py`
    local_model_path = os.path.join(downloaded_path, ARTIFACT_PATH_IN_RUN)
    # ATAU jika downloaded_path sudah mengarah ke folder model:
    # local_model_path = downloaded_path
    
    # Cara terbaik untuk mendapatkan path sebenarnya:
    # downloaded_path itu sendiri sudah path ke folder model (misal: Monitoring_dan_Logging/mlruns_downloaded/walmart_sales_dt_model)
    print(f"\n--- IMPORTANT ---")
    print(f"Your LOCAL MODEL PATH is: {downloaded_path}")
    print(f"--- IMPORTANT ---\n")

except Exception as e:
    logging.error(f"Error downloading artifacts: {e}")
    logging.error(f"Please ensure: ")
    logging.error(f"1. MLflow Tracking Server is running at {mlflow_tracking_uri}")
    logging.error(f"2. Run ID '{TARGET_RUN_ID}' exists in your MLflow Tracking Server.")
    logging.error(f"3. Artifact path '{ARTIFACT_PATH_IN_RUN}' is correct within that run.")