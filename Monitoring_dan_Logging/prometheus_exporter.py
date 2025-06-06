import time
import random
from prometheus_client import start_http_server, Gauge, Counter
import mlflow
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Konfigurasi MLflow Model ---
MLFLOW_MODEL_PATH = "mlruns_downloaded/walmart_sales_dt_model"

# --- Metrik Prometheus ---
INFERENCE_LATENCY = Gauge('model_inference_latency_seconds', 'Latency of model inference in seconds')
INFERENCE_COUNT = Counter('model_inference_total', 'Total number of model inferences')
INFERENCE_ERROR_COUNT = Counter('model_inference_errors_total', 'Total number of model inference errors')
HIGH_SALES_PREDICTION_COUNT = Counter('model_high_sales_predictions_total', 'Total number of predictions exceeding a high sales threshold')
HIGH_SALES_THRESHOLD = 150000.0 

# --- Memuat model MLflow ---
loaded_model = None
try:
    loaded_model = mlflow.pyfunc.load_model(MLFLOW_MODEL_PATH)
    logging.info(f"MLflow model loaded successfully from {MLFLOW_MODEL_PATH}")
except Exception as e:
    logging.error(f"Error loading MLflow model: {e}. Ensure MLFLOW_MODEL_PATH is correct and 'mlruns_downloaded' folder is present.")
    logging.error("Model serving cannot proceed without a loaded model.")

# --- Fungsi untuk Mensimulasikan Inferensi dan Mengumpulkan Metrik ---
def collect_metrics():
    if loaded_model is None:
        logging.warning("Model not loaded, skipping metric collection. Incrementing error count.")
        INFERENCE_ERROR_COUNT.inc()
        return

    # Buat nilai dummy yang realistis untuk setiap fitur
    dummy_data = {
        'Store': [random.randint(1, 45)], 
        'Holiday_Flag': [random.choice([0, 1])],
        'Temperature': [random.uniform(20.0, 80.0)],
        'Fuel_Price': [random.uniform(2.5, 4.5)],
        'CPI': [random.uniform(200.0, 220.0)],
        'Unemployment': [random.uniform(5.0, 10.0)],
        'Month': [random.randint(1, 12)],
        'Year': [random.choice([2010, 2011, 2012])], 
        'Weekly_Sales_Lag1': [random.uniform(50000.0, 200000.0)],
        'Weekly_Sales_Lag2': [random.uniform(50000.0, 200000.0)],
        'Weekly_Sales_Lag4': [random.uniform(50000.0, 200000.0)],
        'Temperature_Fuel_Interaction': [0.0] 
    }

    df_infer = pd.DataFrame(dummy_data)
    
    if 'Temperature' in df_infer.columns and 'Fuel_Price' in df_infer.columns:
        df_infer['Temperature_Fuel_Interaction'] = df_infer['Temperature'] * df_infer['Fuel_Price']
    else:
        logging.warning("Temperature or Fuel_Price not found in dummy_data for interaction term calculation.")


    start_time = time.time()
    try:
        # Melakukan prediksi dengan DataFrame dummy
        prediction = loaded_model.predict(df_infer)
        end_time = time.time()
        latency = end_time - start_time
        
        INFERENCE_LATENCY.set(latency)
        INFERENCE_COUNT.inc()
        
        predicted_sales = prediction[0]
        logging.info(f"Inference successful. Latency: {latency:.4f}s, Predicted Sales: {predicted_sales:.2f}")

        # Metrik kondisional: Jika prediksi penjualan di atas ambang batas
        if predicted_sales > HIGH_SALES_THRESHOLD:
            HIGH_SALES_PREDICTION_COUNT.inc()
            logging.info(f"Predicted sales ({predicted_sales:.2f}) exceeded high sales threshold ({HIGH_SALES_THRESHOLD}).")

    except Exception as e:
        INFERENCE_ERROR_COUNT.inc()
        logging.error(f"Error during inference: {e}")
        logging.error(f"DataFrame causing error:\n{df_infer}")
        logging.error(f"Columns of DataFrame: {df_infer.columns.tolist()}")


if __name__ == '__main__':
    if loaded_model:
        logging.info("Starting Prometheus exporter on port 8000...")
        start_http_server(8000)
        logging.info("Prometheus exporter started. Collecting metrics...")
        
        while True:
            collect_metrics()
            time.sleep(5)
    else:
        logging.critical("Model not loaded, Prometheus exporter will not start.")
