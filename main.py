# main.py
from src.data.load_data import load_fraud_data, load_creditcard_data, load_ip_country_data
from src.data.clean_data import clean_fraud_data
from src.data.feature_engineering import engineer_features_fraud
from src.data.ip_utils import merge_ip_country
from src.data.transform_data import transform_data

import joblib

# Load raw data
fraud_df = load_fraud_data("data/raw/Fraud_Data.csv")
creditcard_df = load_creditcard_data("data/raw/creditcard.csv")
ip_df = load_ip_country_data("data/raw/IpAddress_to_Country.csv")

# Clean and engineer features
fraud_df = clean_fraud_data(fraud_df)
fraud_df = engineer_features_fraud(fraud_df)
fraud_df = merge_ip_country(fraud_df, ip_df)

# Transform data
X_resampled, y_resampled = transform_data(fraud_df)

# Save transformed dataset
joblib.dump((X_resampled, y_resampled), "data/processed/fraud_features_resampled.pkl")
