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
    local_model_path = os.path.join(downloaded_path, ARTIFACT_PATH_IN_RUN)
    print(f"\n--- IMPORTANT ---")
    print(f"Your LOCAL MODEL PATH is: {downloaded_path}")
    print(f"--- IMPORTANT ---\n")

except Exception as e:
    logging.error(f"Error downloading artifacts: {e}")
    logging.error(f"Please ensure: ")
    logging.error(f"1. MLflow Tracking Server is running at {mlflow_tracking_uri}")
    logging.error(f"2. Run ID '{TARGET_RUN_ID}' exists in your MLflow Tracking Server.")
    logging.error(f"3. Artifact path '{ARTIFACT_PATH_IN_RUN}' is correct within that run.")
