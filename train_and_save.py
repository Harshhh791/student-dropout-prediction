import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split

# ----------------------------
# Feature engineering function
# ----------------------------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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

    return df


# ----------------------------
# Load data + engineer features
# ----------------------------
df_raw = pd.read_csv("data/data.csv", sep=";")
df = make_features(df_raw)

# Target mapping
target_map = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
inv_target_map = {v: k for k, v in target_map.items()}

y = df["Target"].map(target_map).astype(int)
X = df.drop(columns=["Target"])

# ----------------------------
# Decide which columns are categorical (for dropdowns)
# Rule of thumb: integer-like + few unique values
# ----------------------------
categorical_cols = []
for col in X.columns:
    nun = X[col].nunique(dropna=True)
    if pd.api.types.is_integer_dtype(X[col]) and nun <= 50:
        categorical_cols.append(col)

# Options for dropdowns
categorical_options = {col: sorted(X[col].dropna().unique().tolist()) for col in categorical_cols}

# Default values (medians) for all columns
feature_defaults = {}
for col in X.columns:
    if col in categorical_cols:
        # default to most common value (mode)
        feature_defaults[col] = int(X[col].mode(dropna=True).iloc[0])
    else:
        # default to median
        feature_defaults[col] = float(X[col].median())

# ----------------------------
# Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Train XGBoost
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
# Save everything needed for the app
# ----------------------------
artifact = {
    "model": model,
    "feature_columns": list(X.columns),
    "target_map": target_map,
    "inv_target_map": inv_target_map,
    "categorical_cols": categorical_cols,
    "categorical_options": categorical_options,
    "feature_defaults": feature_defaults
}

joblib.dump(artifact, "model_artifact.pkl")
print("Saved: model_artifact.pkl")
print("Categorical columns (dropdowns):", len(categorical_cols))
