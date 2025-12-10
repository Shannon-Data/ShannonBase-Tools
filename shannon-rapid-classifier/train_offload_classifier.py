import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import joblib

# ========== 1. read data ==========
df = pd.read_csv("mysql_offload_balanced_5000_IS_OLAP.csv")

X = df.drop(columns=["IS_OLAP"])
y = df["IS_OLAP"]

print("number of samples:", len(df))
print("OLAP ratio:", y.mean())

# ========== 2. training dataset / test dataset ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("size of tranning dataset:", len(X_train))
print("size of testing dataset:", len(X_test))

# ========== 3. define LightGBM classifier model ==========
model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=64,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ========== 4. training ==========
model.fit(X_train, y_train)

# ========== 5. prediction and evaluation ==========
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n====== result of model ======")
print("Accuracy:", acc)
print("AUC:", auc)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ========== 6. importance of features ==========
feat_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n====== Importance Top 10 ======")
print(feat_imp.head(10))

# ========== 7. save as the model file ==========
joblib.dump(model, "rapid_offload_classifiler_model.pkl")
print("\nsave as: rapid_offload_classifier_model.pkl")

