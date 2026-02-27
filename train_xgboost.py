import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("data/data.csv", sep=";")

# -------- Feature engineering (same as before) --------
df["approval_ratio_1st"] = df["Curricular units 1st sem (approved)"] / df["Curricular units 1st sem (enrolled)"].replace(0, 1)
df["approval_ratio_2nd"] = df["Curricular units 2nd sem (approved)"] / df["Curricular units 2nd sem (enrolled)"].replace(0, 1)
df["approval_trend"] = df["Curricular units 2nd sem (approved)"] - df["Curricular units 1st sem (approved)"]
df["enrollment_trend"] = df["Curricular units 2nd sem (enrolled)"] - df["Curricular units 1st sem (enrolled)"]
df["evaluation_ratio_1st"] = df["Curricular units 1st sem (evaluations)"] / df["Curricular units 1st sem (enrolled)"].replace(0, 1)
df["evaluation_ratio_2nd"] = df["Curricular units 2nd sem (evaluations)"] / df["Curricular units 2nd sem (enrolled)"].replace(0, 1)

for col in [
    "approval_ratio_1st",
    "approval_ratio_2nd",
    "evaluation_ratio_1st",
    "evaluation_ratio_2nd"
]:
    df[col] = df[col].clip(0, 1)

for col in ["Debtor", "Tuition fees up to date", "Scholarship holder"]:
    df[col] = df[col].astype(int)

drop_cols = [
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 2nd sem (evaluations)"
]
df = df.drop(columns=drop_cols)

# -------- Target encoding --------
target_map = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
df["Target_encoded"] = df["Target"].map(target_map).astype(int)
df = df.drop(columns=["Target"])

X = df.drop(columns=["Target_encoded"])
y = df["Target_encoded"]

# ----------------------------
# Train-test split (stratified)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# XGBoost model
# ----------------------------
model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
pred = model.predict(X_test)

print("\n=== XGBoost Results ===")
print(classification_report(y_test, pred, target_names=["Dropout","Enrolled","Graduate"]))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
