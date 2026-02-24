# 04 - Modeling
# Objective:
# Build and evaluate fraud detection models under
# severe class imbalance (~0.96% fraud rate).
#
# Key focus:
# - Handling imbalanced data
# - Threshold tuning
# - Precision–Recall tradeoff
# - Model comparison (Linear vs Non-linear)
# - Investigating data-driven performance limitations
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("transactions_clean.csv")

#
##making disputed column
df["is_disputed"] = (df["status"] == "disputed").astype(int)


print("Rows:", len(df))
print("Target distribution:")
print(df["is_disputed"].value_counts())
print("\nTarget rate (%):", df["is_disputed"].mean() * 100)

##Feature selection
# Selected transactional and categorical features.
# Both numerical and categorical variables included.
# Categorical variables will be encoded using one-hot encoding.

feature_cols = [
    "risk_score",
    "amount",
    "country_mismatch",
    "ip_country_missing",
    "device_type_missing",
    "merchant_category_missing",
    "payment_method_missing",
    "merchant_category",
    "device_type",
    "payment_method"
]

X = df[feature_cols]
y = df["is_disputed"]
print("Feature shape:", X.shape)
#Categorical Encoding
X_encoded = pd.get_dummies(
    X,
    columns=["merchant_category", "device_type", "payment_method"],
    drop_first=True  # dummy trap'ı önlemek için
)
print("Encoded feature shape:", X_encoded.shape)
#print(X_encoded.columns)


# Stratified split applied to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=0,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

print("\nTrain disputed distribution:")
print(y_train.value_counts())

print("\nTest disputed distribution:")
print(y_test.value_counts())

# Observation:
# ROC-AUC close to 0.5 indicates weak class separability.
# High false positive rate observed when recall is improved.
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",  # class 1 gets 100 fold weight
    random_state=0
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
 
print("CONFUSION MATRIX")                       # [[TN  FP]
print(confusion_matrix(y_test, y_pred))         #  [FN  TP]]
 
print("\nCLASSIFICATION REPORT")                #recall = TP / (TP + FN) = how many real fraud we caught (% ) |  precision = TP / (TP + FP)= haw many of fraud are really fraud (%) # f1-score = Precision ve recall
print(classification_report(y_test, y_pred, digits=4))

print("\nROC-AUC:", roc_auc_score(y_test, y_prob))


#threshold optimazation
# Default threshold (0.5) may not align with business goals.
# Different probability cutoffs tested to analyze
# recall–precision tradeoff.
thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
y_prob = model.predict_proba(X_test)[:, 1]

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    print(f"\n--- threshold={t} ---")
    print("TN FP FN TP:", tn, fp, fn, tp)
    print(classification_report(y_test, y_pred_t, digits=4))

#Top-K Strategy
k = 200  # günde 200 inceleme diyelim
idx = np.argsort(-y_prob)[:k]  # en yüksek olasılıklar
y_pred_topk = np.zeros_like(y_test.values)
y_pred_topk[idx] = 1
print(confusion_matrix(y_test, y_pred_topk))
print(classification_report(y_test, y_pred_topk, digits=4))

# Random Forest 
# Tested a tree-based ensemble model to capture
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]

print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, digits=4))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))


# Feature Distribution Analysis
# To investigate poor model performance,
# class-wise feature distributions were analyzed.
df.groupby("is_disputed")["risk_score"].describe()
df.groupby("is_disputed")["amount"].describe()

plt.hist(df[df["is_disputed"]==0]["risk_score"], bins=50, alpha=0.5, label="0")
plt.hist(df[df["is_disputed"]==1]["risk_score"], bins=50, alpha=0.5, label="1")
plt.legend()
plt.show()

print(df.groupby("is_disputed")["risk_score"].mean())
print(df.groupby("is_disputed")["amount"].mean())

print(df.groupby("is_disputed")["country_mismatch"].mean())
plt.hist(df[df["is_disputed"]==0]["amount"], bins=50, alpha=0.5, label="0")
plt.hist(df[df["is_disputed"]==1]["amount"], bins=50, alpha=0.5, label="1")
plt.legend()
plt.show()

# Final Insight
# Both linear and non-linear models achieved ROC-AUC ≈ 0.52–0.54.
# This is close to random guessing.
# The main limitation appears to be weak feature signal
# rather than model selection.




