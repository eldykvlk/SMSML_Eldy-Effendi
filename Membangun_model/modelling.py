import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import logging
import numpy as np 

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model():
    logging.info("Starting model training process.")
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000") 
    logging.info("MLflow tracking URI set to http://127.0.0.1:5000.")

    # Nama eksperimen MLflow
    experiment_name = "Walmart Sales Prediction Basic"
    mlflow.set_experiment(experiment_name)
    logging.info(f"MLflow experiment set to: {experiment_name}")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info(f"MLflow Run ID for this run: {run_id}")

        # Mengaktifkan autologging untuk Scikit-learn
        mlflow.sklearn.autolog()
        logging.info("MLflow autologging for scikit-learn enabled.")

        try:
            # Memuat dataset yang telah dipreprocessing
            df = pd.read_csv('Walmart_Sales_preprocessing.csv')
            logging.info("Dataset 'Walmart_Sales_preprocessing.csv' loaded successfully.")
        except FileNotFoundError:
            logging.error("Error: 'Walmart_Sales_preprocessing.csv' not found. Make sure it's in the same directory as modelling.py.")
            return
        except Exception as e:
            logging.error(f"An error occurred while loading the dataset: {e}")
            return

        # Feature Engineering
        # Pastikan kolom 'Date' ada dan dikonversi jika belum (jika preprocessing tidak menyimpan Date)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
            df = df.drop('Date', axis=1) 
            logging.info("Date column processed for Month and Year features.")
        else:
            logging.warning("Date column not found. Ensure preprocessing output includes it if needed for feature engineering.")

        # Create lagged features - Perlu grouping by 'Store'
        if 'Store' in df.columns and 'Weekly_Sales' in df.columns:
            df['Weekly_Sales_Lag1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
            df['Weekly_Sales_Lag2'] = df.groupby('Store')['Weekly_Sales'].shift(2)
            df['Weekly_Sales_Lag4'] = df.groupby('Store')['Weekly_Sales'].shift(4)

            # Fill NaN values with backward fill per group
            for col in ['Weekly_Sales_Lag1', 'Weekly_Sales_Lag2', 'Weekly_Sales_Lag4']:
                df[col] = df.groupby('Store')[col].bfill()
            logging.info("Lagged sales features created and NaNs filled with bfill.")
        else:
            logging.warning("Store or Weekly_Sales column not found for lagged feature creation. Skipping this step.")

        # Create interaction terms
        if 'Temperature' in df.columns and 'Fuel_Price' in df.columns:
            df['Temperature_Fuel_Interaction'] = df['Temperature'] * df['Fuel_Price']
            logging.info("Temperature_Fuel_Interaction feature created.")
        else:
            logging.warning("Temperature or Fuel_Price column not found for interaction feature creation. Skipping this step.")

        # Drop rows with any remaining NaN values (e.g., from lagged features at the beginning of series)
        initial_rows_after_fe = df.shape[0]
        df.dropna(inplace=True)
        if df.shape[0] < initial_rows_after_fe:
            logging.warning(f"Dropped {initial_rows_after_fe - df.shape[0]} rows due to remaining NaN values after feature engineering.")
        logging.info(f"Dataset shape after dropping remaining NaNs: {df.shape}")

        # Definisikan fitur (X) dan variabel target (y)
        # Hapus kolom yang bukan fitur
        features = [col for col in df.columns if col not in ['Weekly_Sales']]
        X = df[features]
        y = df['Weekly_Sales']
        logging.info(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
        logging.info(f"Features used for training: {features}")

        # Bagi data menjadi training dan testing sets
        if 'Store' in df.columns and len(df['Store'].unique()) > 1: # Check if 'Store' exists and has multiple unique values for stratification
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['Store'])
            logging.info("Data split into training and testing sets with stratification on 'Store' column.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.warning("Store column not found or has too few unique values for stratification. Data split without stratification.")

        logging.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

        # Inisialisasi dan latih model DecisionTreeRegressor
        model = DecisionTreeRegressor(random_state=42)
        logging.info("DecisionTreeRegressor model initialized.")
        model.fit(X_train, y_train)
        logging.info("Model training completed.")

        # Buat prediksi pada data uji
        y_pred = model.predict(X_test)
        logging.info("Predictions made on test data.")

        # Evaluasi model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse) # Menggunakan np.sqrt dari numpy
        r2 = r2_score(y_test, y_pred)

        # Log metrik menggunakan MLflow (autolog akan melakukannya, tapi ini untuk eksplisit)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        logging.info(f"Model Metrics logged: MAE={mae:.2f}, MSE={mse:.2f}, RMSE={rmse:.2f}, R2 Score={r2:.2f}")

        # Log model secara eksplisit
        mlflow.sklearn.log_model(model, "walmart_sales_dt_model")
        logging.info("Model logged to MLflow as 'walmart_sales_dt_model'.")

        logging.info("Training process finished.")
        logging.info(f"MLflow Run completed. View results at: mlflow ui --backend-store-uri 'file:{mlflow.get_tracking_uri().replace('http://127.0.0.1:5000', './mlruns')}'")


if __name__ == "__main__":
    train_model()
