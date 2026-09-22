"""
Heart Disease Prediction App
Full-stack: ML backend (scikit-learn) + Frontend (Streamlit)
Dataset: heart.csv (UCI Heart Disease Dataset)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_score, recall_score, f1_score
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        color: #c0392b;
        text-align: center;
        padding: 1rem 0 0.3rem 0;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #fff5f5;
        border-left: 5px solid #c0392b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .predict-positive {
        background: #fdecea;
        border: 2px solid #e74c3c;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .predict-negative {
        background: #eafaf1;
        border: 2px solid #27ae60;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BACKEND — DATA & MODEL
# ─────────────────────────────────────────────

DATASET_PATH = "heart.csv"
MODEL_PATH   = "heart_model.joblib"
SCALER_PATH  = "heart_scaler.joblib"

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

FEATURE_LABELS = {
    "age":      "Age (years)",
    "sex":      "Sex",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol":     "Serum Cholesterol (mg/dl)",
    "fbs":      "Fasting Blood Sugar > 120 mg/dl",
    "restecg":  "Resting ECG Results",
    "thalach":  "Max Heart Rate Achieved",
    "exang":    "Exercise-Induced Angina",
    "oldpeak":  "ST Depression (Exercise vs Rest)",
    "slope":    "Slope of Peak Exercise ST Segment",
    "ca":       "Number of Major Vessels (0–3)",
    "thal":     "Thalassemia Type",
}

@st.cache_data
def load_data():
    df = pd.read_csv(DATASET_PATH)
    return df

@st.cache_resource
def train_model():
    """Train all models once; reload from disk on subsequent restarts."""
    df = pd.read_csv(DATASET_PATH)
    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── If saved models exist, load them instead of retraining ──
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        scaler     = joblib.load(SCALER_PATH)
        X_test_sc  = scaler.transform(X_test)

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM":                 SVC(probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_sc, y_train)
        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]
        results[name] = {
            "model":     model,
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_proba),
            "y_pred":    y_pred,
            "y_proba":   y_proba,
            "cm":        confusion_matrix(y_test, y_pred),
        }

    best_name  = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = results[best_name]["model"]

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler,     SCALER_PATH)

    return results, best_name, best_model, scaler, X_test, y_test

def predict_patient(model, scaler, input_dict):
    input_df = pd.DataFrame([input_dict], columns=FEATURE_COLS)
    input_sc = scaler.transform(input_df)
    pred     = model.predict(input_sc)[0]
    proba    = model.predict_proba(input_sc)[0]
    return int(pred), float(proba[1])

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Heart_anterior_exterior_view.jpg/220px-Heart_anterior_exterior_view.jpg",
    use_column_width=True
)
st.sidebar.markdown("## ❤️ Heart Disease Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Data Explorer", "🤖 Model Performance", "🩺 Predict"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Dataset:** [UCI Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  \n"
    "**Rows:** 1,025 | **Features:** 13  \n"
    "**Target:** 0 = No Disease, 1 = Disease"
)

# ─────────────────────────────────────────────
# LOAD DATA & TRAIN ONCE
# ─────────────────────────────────────────────

df = load_data()
results, best_name, best_model, scaler, X_test, y_test = train_model()

# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────

if page == "🏠 Home":
    st.markdown('<div class="main-header">❤️ Heart Disease Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning-powered early detection of heart disease risk</div>', unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Records", f"{len(df):,}")
    c2.metric("🔢 Features", len(FEATURE_COLS))
    c3.metric("❤️‍🩹 Disease Cases", f"{df['target'].sum():,}")
    c4.metric("✅ Best Accuracy", f"{max(r['accuracy'] for r in results.values()):.1%}")

    st.markdown("---")

    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.subheader("📌 Project Overview")
        st.markdown("""
        This application uses **supervised machine learning** to predict the likelihood of
        heart disease based on clinical patient parameters.

        **Workflow:**
        1. **Data** is loaded from the UCI Heart Disease dataset (1,025 patients)
        2. **Five ML models** are trained and evaluated automatically
        3. The **best model** (by ROC-AUC) is selected for predictions
        4. Clinicians can enter patient data and receive an **instant risk assessment**

        **Models Compared:**
        - Random Forest · Gradient Boosting · Logistic Regression · SVM · KNN
        """)

    with col_r:
        st.subheader("🎯 Target Distribution")
        target_counts = df["target"].value_counts().reset_index()
        target_counts.columns = ["target", "count"]
        target_counts["label"] = target_counts["target"].map({0: "No Disease", 1: "Heart Disease"})
        fig_pie = px.pie(
            target_counts, values="count", names="label",
            color="label",
            color_discrete_map={"Heart Disease": "#e74c3c", "No Disease": "#27ae60"},
            hole=0.45
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False, height=280)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")
    st.subheader("📖 Feature Reference")
    feat_df = pd.DataFrame({
        "Feature": list(FEATURE_LABELS.keys()),
        "Description": list(FEATURE_LABELS.values()),
        "Type": [
            "Numeric","Binary","Categorical","Numeric","Numeric",
            "Binary","Categorical","Numeric","Binary","Numeric",
            "Categorical","Numeric","Categorical"
        ]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ─────────────────────────────────────────────

elif page == "📊 Data Explorer":
    st.markdown('<div class="main-header">📊 Data Explorer</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🗃️ Raw Data", "📈 Distributions", "🔗 Correlations"])

    # ── Raw Data ──
    with tab1:
        st.subheader("Dataset Preview")
        n = st.slider("Rows to display", 5, 100, 20)
        st.dataframe(df.head(n), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Descriptive Statistics")
            st.dataframe(df.describe().T.round(2), use_container_width=True)
        with col2:
            st.subheader("Missing Values")
            miss = pd.DataFrame({
                "Column": df.columns,
                "Missing": df.isnull().sum().values,
                "% Missing": (df.isnull().mean() * 100).round(2).values
            })
            st.dataframe(miss, use_container_width=True, hide_index=True)
            if df.isnull().sum().sum() == 0:
                st.success("✅ No missing values found!")

    # ── Distributions ──
    with tab2:
        st.subheader("Feature Distributions by Target")
        feat = st.selectbox("Select Feature", FEATURE_COLS, format_func=lambda x: FEATURE_LABELS[x])
        fig_dist = px.histogram(
            df, x=feat, color="target",
            barmode="overlay", nbins=30,
            color_discrete_map={0: "#27ae60", 1: "#e74c3c"},
            labels={"target": "Heart Disease", feat: FEATURE_LABELS[feat]},
            opacity=0.75
        )
        fig_dist.update_layout(legend_title_text="0 = No Disease | 1 = Disease", height=380)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader("Age vs Max Heart Rate")
        fig_scatter = px.scatter(
            df, x="age", y="thalach",
            color=df["target"].astype(str),
            color_discrete_map={"0": "#27ae60", "1": "#e74c3c"},
            labels={"age": "Age", "thalach": "Max Heart Rate",
                    "color": "Disease (0=No, 1=Yes)"},
            opacity=0.7
        )
        # manual regression line (no statsmodels needed)
        m, b = np.polyfit(df["age"], df["thalach"], 1)
        x_line = np.linspace(df["age"].min(), df["age"].max(), 100)
        fig_scatter.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode="lines", name="Trend",
            line=dict(color="gray", dash="dash", width=2)
        ))
        fig_scatter.update_layout(height=380)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Correlations ──
    with tab3:
        st.subheader("Correlation Heatmap")
        corr = df.corr(numeric_only=True)
        fig_corr, ax = plt.subplots(figsize=(11, 8))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5,
            annot_kws={"size": 8}
        )
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
        st.pyplot(fig_corr, use_container_width=True)
        plt.close()

        st.subheader("Correlation with Target")
        corr_target = df.corr(numeric_only=True)["target"].drop("target").sort_values()
        fig_bar = px.bar(
            x=corr_target.values, y=corr_target.index,
            orientation="h",
            color=corr_target.values,
            color_continuous_scale="RdYlGn",
            labels={"x": "Correlation", "y": "Feature"}
        )
        fig_bar.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────

elif page == "🤖 Model Performance":
    st.markdown('<div class="main-header">🤖 Model Performance</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Leaderboard ──
    st.subheader("📊 Model Leaderboard")
    leaderboard = pd.DataFrame([
        {
            "Model": name,
            "Accuracy":  f"{r['accuracy']:.1%}",
            "Precision": f"{r['precision']:.1%}",
            "Recall":    f"{r['recall']:.1%}",
            "F1-Score":  f"{r['f1']:.1%}",
            "ROC-AUC":   f"{r['roc_auc']:.1%}",
            "🏆": "✅ Best" if name == best_name else ""
        }
        for name, r in results.items()
    ])
    st.dataframe(leaderboard, use_container_width=True, hide_index=True)
    st.success(f"🏆 Best model selected: **{best_name}** (ROC-AUC = {results[best_name]['roc_auc']:.1%})")

    st.markdown("---")

    # ── ROC Curves ──
    st.subheader("📈 ROC Curves — All Models")
    fig_roc = go.Figure()
    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(dash="dot", color="gray"))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    for (name, r), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={r['roc_auc']:.3f})",
            line=dict(color=color, width=2)
        ))
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420,
        legend=dict(x=0.55, y=0.1)
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # ── Confusion Matrix + Feature Importance ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader(f"🔲 Confusion Matrix — {best_name}")
        cm = results[best_name]["cm"]
        fig_cm, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Reds", ax=ax,
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
            linewidths=1
        )
        ax.set_xlabel("Predicted", fontweight="bold")
        ax.set_ylabel("Actual",    fontweight="bold")
        ax.set_title(f"Confusion Matrix — {best_name}", fontweight="bold")
        st.pyplot(fig_cm, use_container_width=True)
        plt.close()

    with col_b:
        st.subheader("🌟 Feature Importances (Random Forest)")
        rf_model = results["Random Forest"]["model"]
        importance_df = pd.DataFrame({
            "Feature":    FEATURE_COLS,
            "Importance": rf_model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            importance_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance", color_continuous_scale="Reds",
        )
        fig_imp.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # ── Cross-Validation ──
    st.subheader("🔄 5-Fold Cross-Validation (Best Model)")
    with st.spinner("Running cross-validation…"):
        X_all = df[FEATURE_COLS]
        y_all = df["target"]
        X_all_sc = scaler.transform(X_all)
        cv_scores = cross_val_score(
            results[best_name]["model"], X_all_sc, y_all,
            cv=5, scoring="roc_auc"
        )
    cv_df = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(5)], "ROC-AUC": cv_scores.round(4)})
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(cv_df, use_container_width=True, hide_index=True)
        st.info(f"Mean ROC-AUC: **{cv_scores.mean():.4f}** ± {cv_scores.std():.4f}")
    with c2:
        fig_cv = px.bar(
            cv_df, x="Fold", y="ROC-AUC",
            color="ROC-AUC", color_continuous_scale="Reds",
            range_y=[0.8, 1.0]
        )
        fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="gray",
                         annotation_text=f"Mean = {cv_scores.mean():.4f}")
        fig_cv.update_layout(height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_cv, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────

elif page == "🩺 Predict":
    st.markdown('<div class="main-header">🩺 Patient Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Using best model: <b>{best_name}</b> (ROC-AUC = {results[best_name]["roc_auc"]:.1%})</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.form("prediction_form"):
        st.subheader("Enter Patient Clinical Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            age      = st.slider("Age (years)", 20, 90, 55)
            sex      = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp       = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                                    format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina",
                                                            2: "Non-anginal Pain", 3: "Asymptomatic"}[x])
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
            chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 245)

        with col2:
            fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1],
                                    format_func=lambda x: "Yes" if x == 1 else "No")
            restecg  = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                                    format_func=lambda x: {0: "Normal",
                                                            1: "ST-T Wave Abnormality",
                                                            2: "Left Ventricular Hypertrophy"}[x])
            thalach  = st.slider("Max Heart Rate Achieved", 60, 210, 150)
            exang    = st.selectbox("Exercise-Induced Angina", options=[0, 1],
                                    format_func=lambda x: "Yes" if x == 1 else "No")

        with col3:
            oldpeak  = st.slider("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
            slope    = st.selectbox("Slope of Peak ST Segment", options=[0, 1, 2],
                                    format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
            ca       = st.selectbox("Major Vessels Colored (0–3)", options=[0, 1, 2, 3])
            thal     = st.selectbox("Thalassemia", options=[0, 1, 2, 3],
                                    format_func=lambda x: {0: "Normal", 1: "Fixed Defect",
                                                            2: "Reversible Defect", 3: "Unknown"}[x])

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        input_data = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
            "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
        }
        pred, prob = predict_patient(best_model, scaler, input_data)

        st.markdown("---")
        st.subheader("🔬 Prediction Result")

        r_col1, r_col2, r_col3 = st.columns([1, 1.5, 1])

        with r_col2:
            if pred == 1:
                st.markdown(f"""
                <div class="predict-positive">
                    <h2 style="color:#e74c3c">⚠️ Heart Disease Detected</h2>
                    <p style="font-size:1.2rem">Risk Probability: <b>{prob:.1%}</b></p>
                    <p style="color:#555">Please consult a cardiologist immediately.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="predict-negative">
                    <h2 style="color:#27ae60">✅ No Heart Disease Detected</h2>
                    <p style="font-size:1.2rem">Risk Probability: <b>{prob:.1%}</b></p>
                    <p style="color:#555">Continue regular health monitoring.</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 Risk Gauge")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            delta={"reference": 50, "suffix": "%"},
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": "#e74c3c" if pred == 1 else "#27ae60"},
                "steps": [
                    {"range": [0,  30], "color": "#d5f5e3"},
                    {"range": [30, 60], "color": "#fdebd0"},
                    {"range": [60,100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line":  {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": 50
                }
            },
            title={"text": "Heart Disease Risk (%)"}
        ))
        fig_gauge.update_layout(height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("📝 Input Summary")
        summary_df = pd.DataFrame(
            {"Parameter": list(FEATURE_LABELS.values()),
             "Value":     list(input_data.values())}
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.markdown("""
        ---
        > ⚕️ **Medical Disclaimer:** This tool is for educational and research purposes only.
        > It is **not** a substitute for professional medical advice, diagnosis, or treatment.
        """)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:0.85rem'>"
    "❤️ Heart Disease Prediction System &nbsp;|&nbsp; "
    "Dataset: UCI Heart Disease &nbsp;|&nbsp; Built with Streamlit & scikit-learn"
    "</div>",
    unsafe_allow_html=True
)
