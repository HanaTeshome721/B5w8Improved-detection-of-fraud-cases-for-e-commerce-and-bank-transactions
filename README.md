# Task 1 – Data Analysis and Preprocessing

This module is part of the **Fraud Detection for E-commerce and Bank Transactions** project at **Adey Innovations Inc.** It focuses on analyzing and preparing raw data for machine learning by cleaning, enriching, transforming, and understanding the fraud transaction data.

---

##  Objective

Prepare the `Fraud_Data.csv` and `IpAddress_to_Country.csv` datasets for modeling by:

- Handling missing values and duplicates
- Performing Exploratory Data Analysis (EDA)
- Engineering features to extract time and behavior signals
- Merging geolocation data using IP ranges
- Addressing class imbalance
- Encoding and scaling the data

---

## 📁 Folder Structure

fraud-detection-project/
│
├── data/
│ ├── raw/ # Original CSV files
│ ├── interim/ # Cleaned but untransformed data
│ └── processed/ # Final dataset after all transformations
│
├── notebooks/
│ └── 01_data_preprocessing.ipynb # EDA & preprocessing notebook
│
├── src/
│ └── data/
│ ├── load_data.py
│ ├── clean_data.py
│ ├── feature_engineering.py
│ ├── ip_utils.py
│ └── transform_data.py
│
│
├── main.py # Run end-to-end preprocessing
└── README.md # This file

yaml
Copy
Edit

---

## 📊 Datasets Used

1. **Fraud_Data.csv**
   - E-commerce transactions labeled as fraudulent or not
   - Features include user/device metadata, browser, purchase value, signup time, etc.

2. **IpAddress_to_Country.csv**
   - IP address ranges mapped to country names

---

## ⚙️ Task Breakdown

### ✅ Step 1: Load and Inspect Data
- Read CSVs using pandas
- Check datatypes, missing values, and basic statistics

### ✅ Step 2: Data Cleaning
- Remove duplicates
- Fix data types (e.g., datetime, numeric conversions)
- Drop or impute missing values

### ✅ Step 3: EDA (Exploratory Data Analysis)
- Visualize class imbalance
- Distributions of age, purchase value, etc.
- Boxplots and correlation heatmaps

### ✅ Step 4: Feature Engineering
- Time-based features: `hour_of_day`, `day_of_week`, `time_since_signup`
- Behavior-based: `transaction_count` per user
- Geolocation: Convert IP to country via range mapping

### ✅ Step 5: Data Transformation
- Normalize numeric features using `StandardScaler`
- Encode categorical features using One-Hot Encoding
- Apply SMOTE to handle class imbalance

---

## 🧪 Outputs

- Cleaned and transformed dataset:
data/processed/fraud_features_resampled.pkl

diff

- Visualizations:
outputs/eda_visuals/

css

- Summary Notebook:
notebooks/01_data_preprocessing.ipynb

yaml


---

## 📦 Required Packages

Install required dependencies:

```bash
pip install -r requirements.txt
📈 Next Steps
Proceed to Task 2 – Feature Selection and Model Training, where you'll use the processed data to train ML models and evaluate fraud prediction performance.
