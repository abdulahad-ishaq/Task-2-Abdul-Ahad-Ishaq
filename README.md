# Task-2-Abdul-Ahad-Ishaq
# Project 2: Fraud Detection Pipeline

Trains and tunes classifiers to flag suspicious orders in a highly imbalanced dataset, using a leak-free SMOTE pipeline and evaluating on Precision, Recall, and ROC-AUC instead of Accuracy.

## Label

The source dataset has no fraud column, so IsFraud is derived from a risk score (price/quantity mismatch, per-product price outliers, cart abandonment, risky payment + cancellation). An order needs 2+ signals to be flagged, keeping the positive class rare and realistic. Swap in a real labeled target if you have one, the rest of the pipeline is unaffected.

## Pipeline

- Split first. train_test_split runs before any resampling or scaling touches the data.
- imblearn.pipeline.Pipeline, not sklearn's, SMOTE is applied only inside each training fold, never to the validation fold or test set.
- Logistic Regression: StandardScaler → SMOTE → LogisticRegression
- Random Forest: SMOTE → RandomForestClassifier (tree splits are scale-invariant, so no scaler)
- Tuning: GridSearchCV (5-fold stratified, roc_auc scoring) over both smote__k_neighbors and the model's own hyperparameters jointly.
- Evaluation: Precision, Recall, ROC-AUC, confusion matrix, Accuracy is never reported.

## Setup

pip install pandas numpy scikit-learn imbalanced-learn

Place the source CSV at data/ecommerce_orders.csv (same schema as Project 1).

## Run

python fraud_detection_pipeline.py

Prints fraud rate, best params per model, and a final Precision/Recall/ROC-AUC comparison.
