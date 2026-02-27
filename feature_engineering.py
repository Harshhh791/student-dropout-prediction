import pandas as pd

df = pd.read_csv("data/data.csv", sep=";")
df["approval_ratio_1st"] = (
    df["Curricular units 1st sem (approved)"] /
    df["Curricular units 1st sem (enrolled)"].replace(0, 1)
)

df["approval_ratio_2nd"] = (
    df["Curricular units 2nd sem (approved)"] /
    df["Curricular units 2nd sem (enrolled)"].replace(0, 1)
)
df["approval_trend"] = (
    df["Curricular units 2nd sem (approved)"] -
    df["Curricular units 1st sem (approved)"]
)

df["enrollment_trend"] = (
    df["Curricular units 2nd sem (enrolled)"] -
    df["Curricular units 1st sem (enrolled)"]
)
df["evaluation_ratio_1st"] = (
    df["Curricular units 1st sem (evaluations)"] /
    df["Curricular units 1st sem (enrolled)"].replace(0, 1)
)

df["evaluation_ratio_2nd"] = (
    df["Curricular units 2nd sem (evaluations)"] /
    df["Curricular units 2nd sem (enrolled)"].replace(0, 1)
)

ratio_cols = [
    "approval_ratio_1st",
    "approval_ratio_2nd",
    "evaluation_ratio_1st",
    "evaluation_ratio_2nd"
]

for col in ratio_cols:
    df[col] = df[col].clip(0, 1)

    
risk_cols = [
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder"
]

for col in risk_cols:
    df[col] = df[col].astype(int)

drop_cols = [
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 2nd sem (evaluations)"
]

df_model = df.drop(columns=drop_cols)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_model["Target_encoded"] = le.fit_transform(df_model["Target"])

df_model = df_model.drop(columns=["Target"])

print(dict(zip(le.classes_, le.transform(le.classes_))))

print(df_model.shape)
print(df_model.head())
print(df_model.isna().sum().sum())

label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
label_mapping = {k: int(v) for k, v in label_mapping.items()}
print("Label mapping:", label_mapping)
