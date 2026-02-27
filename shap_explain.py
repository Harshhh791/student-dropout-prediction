import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ----------------------------
# 1) Load data
# ----------------------------
df = pd.read_csv("data/data.csv", sep=";")

# ----------------------------
# 2) Feature engineering (same as your pipeline)
# ----------------------------
df["approval_ratio_1st"] = df["Curricular units 1st sem (approved)"] / df["Curricular units 1st sem (enrolled)"].replace(0, 1)
df["approval_ratio_2nd"] = df["Curricular units 2nd sem (approved)"] / df["Curricular units 2nd sem (enrolled)"].replace(0, 1)
df["approval_trend"] = df["Curricular units 2nd sem (approved)"] - df["Curricular units 1st sem (approved)"]
df["enrollment_trend"] = df["Curricular units 2nd sem (enrolled)"] - df["Curricular units 1st sem (enrolled)"]
df["evaluation_ratio_1st"] = df["Curricular units 1st sem (evaluations)"] / df["Curricular units 1st sem (enrolled)"].replace(0, 1)
df["evaluation_ratio_2nd"] = df["Curricular units 2nd sem (evaluations)"] / df["Curricular units 2nd sem (enrolled)"].replace(0, 1)

for col in ["approval_ratio_1st", "approval_ratio_2nd", "evaluation_ratio_1st", "evaluation_ratio_2nd"]:
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

# ----------------------------
# 3) Target encoding (same mapping)
# ----------------------------
target_map = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
df["Target_encoded"] = df["Target"].map(target_map).astype(int)
df = df.drop(columns=["Target"])

X = df.drop(columns=["Target_encoded"])
y = df["Target_encoded"]

# ----------------------------
# 4) Train/test split (stratified)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 5) Train XGBoost for SHAP
# IMPORTANT: use "multi:softprob" so SHAP can explain probabilities
# ----------------------------
model = xgb.XGBClassifier(
    objective="multi:softprob",
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
# 6) SHAP Explainer
# ----------------------------
explainer = shap.TreeExplainer(model)

# explain a sample for speed
X_explain = X_test.sample(n=min(600, len(X_test)), random_state=42)

shap_values = explainer.shap_values(X_explain)

class_names = ["Dropout", "Enrolled", "Graduate"]

def get_class_shap(shap_values_obj, class_idx: int):
    """
    Handles both SHAP multiclass formats:
    1) list of arrays: shap_values[class_idx] -> (n_samples, n_features)
    2) 3D array: shap_values[:, :, class_idx] -> (n_samples, n_features)
    """
    if isinstance(shap_values_obj, list):
        return shap_values_obj[class_idx]
    arr = shap_values_obj
    # expected: (n_samples, n_features, n_classes)
    return arr[:, :, class_idx]

# ----------------------------
# 7) Global importance per class (bar)
# ----------------------------
for i, cname in enumerate(class_names):
    sv = get_class_shap(shap_values, i)
    shap.summary_plot(sv, X_explain, plot_type="bar", show=True)

# ----------------------------
# 8) Global impact per class (beeswarm)
# ----------------------------
for i, cname in enumerate(class_names):
    sv = get_class_shap(shap_values, i)
    shap.summary_plot(sv, X_explain, show=True)

# ----------------------------
# 9) Local explanation for one student
# ----------------------------
idx = X_explain.index[0]
x_one = X_explain.loc[[idx]]

sv_one = explainer.shap_values(x_one)

# Predicted probs + class
probs = model.predict_proba(x_one)[0]
pred_class = int(np.argmax(probs))
print("\nLocal explanation sample index:", idx)
print("Predicted probabilities:", dict(zip(class_names, probs)))
print("Predicted class:", class_names[pred_class])

# Waterfall for predicted class
try:
    if isinstance(sv_one, list):
        sv_for_class = sv_one[pred_class][0]
    else:
        sv_for_class = sv_one[0, :, pred_class]

    exp = shap.Explanation(
        values=sv_for_class,
        base_values=explainer.expected_value[pred_class] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
        data=x_one.iloc[0],
        feature_names=X.columns
    )
    shap.plots.waterfall(exp, max_display=12)
except Exception as e:
    print("Waterfall plot skipped (SHAP version mismatch):", e)

