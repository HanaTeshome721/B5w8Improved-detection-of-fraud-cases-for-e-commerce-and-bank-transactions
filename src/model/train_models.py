# src/model/train_models.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# Helper Functions
# -------------------

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"===== {model_name} Evaluation =====")
    print(f"F1-Score: {f1:.4f}")
    print(f"Average Precision (AUC-PR): {avg_precision:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

    # Plot PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f'{model_name} (AUC-PR = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return {"f1": f1, "avg_precision": avg_precision, "confusion_matrix": cm}


# -------------------
# Model Training
# -------------------

def train_logistic_regression(X_train, y_train):
    log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')
    log_reg.fit(X_train, y_train)
    return log_reg


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


# -------------------
# Main Pipeline
# -------------------
if __name__ == "__main__":
    # Load processed Fraud Data
    X_fraud, y_fraud = joblib.load("data/processed/fraud_features_resampled.pkl")

    # Train-test split
    X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = split_data(X_fraud, y_fraud)

    # Logistic Regression
    log_reg_fraud = train_logistic_regression(X_train_fraud, y_train_fraud)
    evaluate_model(log_reg_fraud, X_test_fraud, y_test_fraud, "Logistic Regression - Fraud Data")

    # Random Forest
    rf_fraud = train_random_forest(X_train_fraud, y_train_fraud)
    evaluate_model(rf_fraud, X_test_fraud, y_test_fraud, "Random Forest - Fraud Data")

    # Save models
    joblib.dump(log_reg_fraud, "models/logistic_regression_fraud.pkl")
    joblib.dump(rf_fraud, "models/random_forest_fraud.pkl")

    # -------------------
    # CREDITCARD Dataset
    # -------------------
    creditcard_df = pd.read_csv("data/raw/creditcard.csv")
    y_credit = creditcard_df['Class']
    X_credit = creditcard_df.drop(columns=['Class'])

    X_train_cc, X_test_cc, y_train_cc, y_test_cc = split_data(X_credit, y_credit)

    # Logistic Regression (Creditcard)
    log_reg_cc = train_logistic_regression(X_train_cc, y_train_cc)
    evaluate_model(log_reg_cc, X_test_cc, y_test_cc, "Logistic Regression - Credit Card")

    # Random Forest (Creditcard)
    rf_cc = train_random_forest(X_train_cc, y_train_cc)
    evaluate_model(rf_cc, X_test_cc, y_test_cc, "Random Forest - Credit Card")

    # Save models
    joblib.dump(log_reg_cc, "models/logistic_regression_creditcard.pkl")
    joblib.dump(rf_cc, "models/random_forest_creditcard.pkl")
