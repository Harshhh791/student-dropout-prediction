import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

# ----------------------------
# Streamlit page setup
# ----------------------------
st.set_page_config(page_title="Student Dropout Prediction", layout="wide")

# ----------------------------
# Column normalization helpers
# ----------------------------
def norm_col(s: str) -> str:
    # Normalize a single column name (handles tabs, BOM, weird whitespace)
    s = str(s).replace("\ufeff", "").replace("\t", " ").replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split())  # collapse whitespace
    return s.strip()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [norm_col(c) for c in df.columns]
    return df

def normalize_list(cols) -> list:
    return [norm_col(c) for c in cols]

# ----------------------------
# Load saved artifact once
# ----------------------------
@st.cache_resource
def load_artifact():
    return joblib.load("model_artifact.pkl")

artifact = load_artifact()
model = artifact["model"]

# NOTE: feature_cols may contain hidden tabs/spaces from training time
feature_cols = artifact["feature_columns"]
inv_target_map = artifact["inv_target_map"]
categorical_cols = set(artifact.get("categorical_cols", []))
categorical_options = artifact.get("categorical_options", {})
feature_defaults = artifact.get("feature_defaults", {})

# Build normalized schema mapping (normalized -> trained/original)
feature_cols_norm = normalize_list(feature_cols)
norm_to_trained = {n: orig for n, orig in zip(feature_cols_norm, feature_cols)}

class_names = [inv_target_map[i] for i in sorted(inv_target_map.keys())]

# ----------------------------
# Feature engineering for RAW CSV uploads
# (must match your training pipeline)
# ----------------------------
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = normalize_columns(df)

    # Validate required raw columns exist (normalized names)
    required_raw = [
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 2nd sem (evaluations)",
    ]
    missing_raw = [c for c in required_raw if c not in df.columns]
    if missing_raw:
        raise ValueError(f"Missing required raw columns: {missing_raw}")

    # Engineered features
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

    # Clip ratios to [0, 1]
    for col in ["approval_ratio_1st", "approval_ratio_2nd", "evaluation_ratio_1st", "evaluation_ratio_2nd"]:
        df[col] = df[col].clip(0, 1)

    # Binary flags to int (if present)
    for col in ["Debtor", "Tuition fees up to date", "Scholarship holder"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Drop raw columns replaced by engineered ones
    drop_cols = [
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 2nd sem (evaluations)",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df

# ----------------------------
# SHAP helper for multiclass
# ----------------------------
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

def get_class_shap(shap_values_obj, class_idx: int):
    """
    Handles both multiclass SHAP formats:
    1) list of arrays: shap_values[class_idx] -> (n_samples, n_features)
    2) 3D array: shap_values[:, :, class_idx] -> (n_samples, n_features)
    """
    if isinstance(shap_values_obj, list):
        return shap_values_obj[class_idx]
    return shap_values_obj[:, :, class_idx]

# ----------------------------
# App title
# ----------------------------
st.title("🎓 Student Dropout Prediction Dashboard")
st.write("Predict whether a student will **Dropout**, remain **Enrolled**, or **Graduate** using an **XGBoost** model.")

tab1, tab2 = st.tabs(["👤 Single Student", "📄 Batch CSV Predictions"])

# ==========================================================
# TAB 1: Single Student Prediction + SHAP
# ==========================================================
with tab1:
    st.subheader("Single Student Prediction")

    use_typical = st.sidebar.toggle("Use typical (median/mode) defaults", value=True)
    st.sidebar.header("Input Student Features")

    input_data = {}
    for trained_col in feature_cols:
        # UI label should be normalized for readability
        ui_label = norm_col(trained_col)
        default_val = feature_defaults.get(trained_col, 0.0) if use_typical else 0.0

        if trained_col in categorical_cols and trained_col in categorical_options:
            opts = [int(x) for x in categorical_options[trained_col]]
            try:
                default_index = opts.index(int(default_val))
            except Exception:
                default_index = 0
            input_data[trained_col] = st.sidebar.selectbox(ui_label, options=opts, index=default_index)
        else:
            input_data[trained_col] = st.sidebar.number_input(ui_label, value=float(default_val))

    # DataFrame in trained feature order
    X_one_df = pd.DataFrame([input_data], columns=feature_cols)

    # Predict with numpy to avoid any feature-name strictness issues
    probs = model.predict_proba(X_one_df.to_numpy())[0]
    pred_class = int(np.argmax(probs))
    pred_label = inv_target_map[pred_class]

    colA, colB = st.columns([1.05, 1])

    with colA:
        st.markdown("### Prediction")
        st.success(f"Predicted Outcome: **{pred_label}**")

        st.markdown("### Prediction Probabilities")
        prob_df = pd.DataFrame({"Class": class_names, "Probability": probs}).sort_values("Probability", ascending=False)
        st.dataframe(prob_df, use_container_width=True)

    with colB:
        st.markdown("### Why this prediction? (SHAP)")

        explainer = get_explainer(model)

        # SHAP expects dataframe; we pass the same X_one_df
        shap_vals = explainer.shap_values(X_one_df)

        sv_for_class = get_class_shap(shap_vals, pred_class)[0]  # (n_features,)

        contrib = pd.Series(sv_for_class, index=[norm_col(c) for c in feature_cols])
        contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(12)

        fig = plt.figure(figsize=(7, 5))
        plt.barh(contrib.index[::-1], contrib.values[::-1])
        plt.axvline(0, linewidth=1)
        plt.title(f"Top SHAP contributions toward '{pred_label}'")
        plt.xlabel("SHAP value (impact on model output)")
        plt.tight_layout()
        st.pyplot(fig)

        st.caption("Positive bars push toward the predicted class; negative bars push away.")

# ==========================================================
# TAB 2: Batch Prediction from CSV Upload
# ==========================================================
with tab2:
    st.subheader("Batch Predictions (Upload CSV)")
    st.write(
        "Upload a CSV containing **many students**. The app will generate predictions and probabilities for each row, "
        "and you can download the results."
    )

    file_type = st.radio(
        "What type of file are you uploading?",
        ["Raw UCI format (needs feature engineering)", "Already engineered (has model features)"],
        horizontal=True
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    risk_threshold = st.slider("Dropout risk threshold (flag high risk)", 0.0, 1.0, 0.60, 0.01)

    st.info(
        "✅ If your CSV has a `Target` column, we will ignore it. "
        "If uploading RAW format, it must include the curricular units columns used for feature engineering."
    )

    if uploaded is not None:
        raw = pd.read_csv(uploaded, sep=None, engine="python")
        raw = normalize_columns(raw)

        st.write("Normalized columns (uploaded):", raw.columns.tolist())
        st.write("Preview of uploaded data:")
        st.dataframe(raw.head(), use_container_width=True)

        raw_no_target = raw.drop(columns=["Target"], errors="ignore")

        # Prepare features
        try:
            if file_type.startswith("Raw"):
                feat = make_features(raw_no_target)
            else:
                feat = normalize_columns(raw_no_target.copy())
        except Exception as e:
            st.error("Feature preparation failed.")
            st.write("Error:", str(e))
            st.stop()

        # Normalize feat cols and build normalized -> actual map
        feat = normalize_columns(feat)
        feat_cols_norm = normalize_list(feat.columns)
        norm_to_feat = {n: orig for n, orig in zip(feat_cols_norm, feat.columns)}

        # Check missing using normalized comparison against training schema (normalized)
        missing_norm = [n for n in feature_cols_norm if n not in norm_to_feat]
        if missing_norm:
            st.error("Your file is missing required columns for the model (normalized match):")
            st.write(missing_norm)
            st.stop()

        # Build dataframe in TRAINED order (use trained/original col names)
        X_batch_df = pd.DataFrame({
            trained_col: feat[norm_to_feat[n]].values
            for n, trained_col in zip(feature_cols_norm, feature_cols)
        })

        # Predict with numpy to avoid strict name matching in XGBoost
        probs = model.predict_proba(X_batch_df.to_numpy())
        pred_class = np.argmax(probs, axis=1)
        pred_label = [inv_target_map[int(i)] for i in pred_class]

        out = raw.copy()
        out["PredictedOutcome"] = pred_label
        out["P(Dropout)"] = probs[:, 0]
        out["P(Enrolled)"] = probs[:, 1]
        out["P(Graduate)"] = probs[:, 2]
        out["DropoutRiskFlag"] = out["P(Dropout)"] >= risk_threshold

        st.markdown("### Predictions (first 50 rows)")
        st.dataframe(out.head(50), use_container_width=True)

        st.markdown("### Summary")
        st.write(out["PredictedOutcome"].value_counts())
        st.write(f"High-risk students (P(Dropout) ≥ {risk_threshold:.2f}): **{int(out['DropoutRiskFlag'].sum())}**")

        # Download results
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download predictions as CSV",
            data=csv_bytes,
            file_name="dropout_predictions.csv",
            mime="text/csv"
        )
