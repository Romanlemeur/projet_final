
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

BASE = os.path.dirname(os.path.dirname(__file__))
PAIRS_CSV = os.path.join(BASE, "data", "transition_pairs.csv")
OUT_MODEL = os.path.join(BASE, "models", "transition_model_rf.joblib")
OUT_MODEL_LR = os.path.join(BASE, "models", "transition_model_lr.joblib")
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)

df = pd.read_csv(PAIRS_CSV)

# features we use
features = ["delta_bpm", "delta_key", "delta_energy", "ratio_energy"]
X = df[features].values
y = df["label"].values

# split (stratify si possible)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(set(y))>1 else None)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, pred_lr)
f1_lr = f1_score(y_test, pred_lr)
print("LogReg acc:", acc_lr, "f1:", f1_lr)
print(classification_report(y_test, pred_lr))

# RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, pred_rf)
f1_rf = f1_score(y_test, pred_rf)
print("RF acc:", acc_rf, "f1:", f1_rf)
print(classification_report(y_test, pred_rf))

# save the best (choose RF)
joblib.dump(rf, OUT_MODEL)
joblib.dump(lr, OUT_MODEL_LR)
print("Saved models to:", OUT_MODEL, OUT_MODEL_LR)

# print confusion matrix for RF
cm = confusion_matrix(y_test, pred_rf)
print("Confusion matrix (RF):\n", cm)
