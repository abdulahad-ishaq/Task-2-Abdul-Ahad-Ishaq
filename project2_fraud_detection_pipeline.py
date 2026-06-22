"""
DecodeLabs Industrial Training - Data Science Project 2
Supervised Learning: Fraud Detection Pipeline

The source data is the same order-level export used in Project 1.
It doesn't ship with a fraud label, so a label is derived first from a
small set of suspicious-order heuristics (the kind of rules a risk
team would actually flag on: extreme price-quantity mismatches, high
cart abandonment, and risky payment/status combinations). That keeps
the rest of the brief - extreme class imbalance, SMOTE inside a
leak-free imblearn pipeline, GridSearchCV, and Precision/Recall/ROC-AUC
instead of Accuracy - faithful to a real fraud-detection setup.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RAW_PATH = "data/ecommerce_orders.csv"
RANDOM_STATE = 42


def load_data(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["TotalPrice", "Quantity", "UnitPrice"]).reset_index(drop=True)
    return df


def derive_fraud_label(df):
    """
    Flags an order as suspicious when at least two independent risk
    signals line up at once, rather than on a single noisy rule:

      - price/quantity mismatch: TotalPrice is far from Quantity x UnitPrice
      - extreme unit price for its product category (IQR outlier)
      - cart abandonment: very few of the items in the cart were bought
      - a high-risk payment method on an order that was later cancelled

    Two-out-of-four keeps the positive class rare, which is exactly the
    setting SMOTE and Precision/Recall/ROC-AUC are meant for.
    """
    expected_total = df["Quantity"] * df["UnitPrice"]
    price_mismatch = (df["TotalPrice"] - expected_total).abs() / expected_total.replace(0, np.nan) > 0.15

    price_outlier = pd.Series(False, index=df.index)
    for product, group in df.groupby("Product"):
        q1, q3 = group["UnitPrice"].quantile([0.25, 0.75])
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        price_outlier.loc[group.index] = group["UnitPrice"] > upper_fence

    cart_abandon = (df["Quantity"] / df["ItemsInCart"].replace(0, np.nan)) < 0.25

    risky_payment_cancel = df["PaymentMethod"].isin(["Gift Card", "Online"]) & (
        df["OrderStatus"] == "Cancelled"
    )

    risk_score = (
        price_mismatch.astype(int)
        + price_outlier.astype(int)
        + cart_abandon.astype(int)
        + risky_payment_cancel.astype(int)
    )

    df["IsFraud"] = (risk_score >= 2).astype(int)
    return df


def engineer_features(df):
    df["PricePerUnit"] = df["TotalPrice"] / df["Quantity"].replace(0, np.nan)
    df["CartFillRatio"] = (df["Quantity"] / df["ItemsInCart"].replace(0, np.nan)).clip(upper=1.0)
    df["OrderMonth"] = df["Date"].dt.month
    df["OrderDayOfWeek"] = df["Date"].dt.dayofweek
    df["HasCoupon"] = df["CouponCode"].notna().astype(int)
    df = df.dropna(subset=["PricePerUnit", "CartFillRatio"]).reset_index(drop=True)
    return df


def build_model_matrix(df):
    categorical_cols = ["Product", "PaymentMethod", "OrderStatus", "ReferralSource"]
    feature_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    drop_cols = ["OrderID", "CustomerID", "ShippingAddress", "TrackingNumber",
                 "Date", "CouponCode", "IsFraud"]
    X = feature_df.drop(columns=[c for c in drop_cols if c in feature_df.columns])
    y = feature_df["IsFraud"]
    return X, y


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n--- {name} ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    return {"model": name, "precision": precision, "recall": recall, "roc_auc": auc}


def main():
    df = load_data(RAW_PATH)
    df = derive_fraud_label(df)
    df = engineer_features(df)

    fraud_rate = df["IsFraud"].mean()
    print(f"Dataset size: {len(df)} orders")
    print(f"Fraud rate: {fraud_rate:.2%} ({df['IsFraud'].sum()} flagged orders)")

    X, y = build_model_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []

    logistic_pipeline = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    logistic_grid = {
        "smote__k_neighbors": [3, 5],
        "classifier__C": [0.01, 0.1, 1.0],
    }

    logistic_search = GridSearchCV(
        logistic_pipeline, logistic_grid, scoring="roc_auc", cv=cv, n_jobs=-1
    )
    logistic_search.fit(X_train, y_train)
    print(f"\nBest Logistic Regression params: {logistic_search.best_params_}")
    results.append(evaluate("Logistic Regression", logistic_search.best_estimator_, X_test, y_test))

    forest_pipeline = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])

    forest_grid = {
        "smote__k_neighbors": [3, 5],
        "classifier__max_depth": [10, 20, None],
        "classifier__n_estimators": [100, 200],
    }

    forest_search = GridSearchCV(
        forest_pipeline, forest_grid, scoring="roc_auc", cv=cv, n_jobs=-1
    )
    forest_search.fit(X_train, y_train)
    print(f"\nBest Random Forest params: {forest_search.best_params_}")
    results.append(evaluate("Random Forest", forest_search.best_estimator_, X_test, y_test))

    summary = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\nModel comparison (ranked by ROC-AUC):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
