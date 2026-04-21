"""
Telco Customer Churn Dashboard — Streamlit App
================================================
Dataset : WA_Fn-UseC_-Telco-Customer-Churn.csv
Model   : sklearn RandomForestClassifier

Run
---
    pip install streamlit plotly pandas scikit-learn
    streamlit run app.py

Place the CSV file in the same folder as app.py, OR the app will
generate a realistic synthetic dataset automatically so you can
preview the dashboard without the file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── sklearn imports (model section) ──────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── metric cards ── */
[data-testid="metric-container"] {
    background: #f8f7f3;
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="metric-container"] label {
    font-size: 11px !important;
    color: #7a7870 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 500 !important;
    letter-spacing: -0.02em;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 11px !important;
}

/* ── section headers ── */
h1 { font-size: 22px !important; font-weight: 500 !important; letter-spacing: -0.02em !important; }
h2 { font-size: 16px !important; font-weight: 500 !important; letter-spacing: -0.01em !important; }
h3 { font-size: 13px !important; font-weight: 500 !important; color: #7a7870 !important; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #f5f4f0;
    border-right: 1px solid rgba(0,0,0,0.07);
}

/* ── tab bar ── */
[data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid rgba(0,0,0,0.07);
    padding-bottom: 0;
}
[data-baseweb="tab"] {
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 8px 16px !important;
    border-radius: 6px 6px 0 0 !important;
}

/* ── dividers ── */
hr { border-color: rgba(0,0,0,0.06); margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (Plotly-friendly hex)
# ─────────────────────────────────────────────────────────────────────────────
RED    = "#d63b3b"
BLUE   = "#2f7ec4"
GREEN  = "#2e8f5e"
AMBER  = "#c47d0a"
PURPLE = "#6b52c8"
TEAL   = "#1d9e75"
GRAY   = "#888780"

CHURN_COLORS = {True: RED, False: BLUE, "Yes": RED, "No": BLUE}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", size=12, color="#3d3d3a"),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="rgba(0,0,0,0.05)", zeroline=False),
    yaxis=dict(gridcolor="rgba(0,0,0,0.05)", zeroline=False),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="left", x=0, font=dict(size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str | None = None) -> pd.DataFrame:
    """Load real CSV or generate synthetic Telco-like dataset."""
    if path:
        try:
            df = pd.read_csv(path)
            return _clean(df)
        except Exception:
            pass

    # ── synthetic fallback ──
    np.random.seed(42)
    n = 7043
    tenure   = np.random.exponential(scale=32, size=n).clip(1, 72).astype(int)
    monthly  = np.random.normal(64.76, 30, n).clip(18, 119).round(2)
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n, p=[0.55, 0.21, 0.24]
    )
    internet = np.random.choice(
        ["Fiber optic", "DSL", "No"],
        size=n, p=[0.44, 0.34, 0.22]
    )
    payment  = np.random.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n, p=[0.34, 0.23, 0.22, 0.21]
    )
    senior = np.random.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], size=n, p=[0.48, 0.52])

    # churn probability driven by features
    p_churn = (
        0.43 * (contract == "Month-to-month")
        + 0.11 * (contract == "One year")
        + 0.03 * (contract == "Two year")
        + 0.10 * (internet == "Fiber optic")
        + 0.08 * (payment == "Electronic check")
        + 0.08 * (senior == 1)
        - 0.01 * (tenure / 72)
    )
    p_churn = (p_churn / p_churn.max() * 0.65).clip(0, 0.95)
    churn   = np.where(np.random.rand(n) < p_churn, "Yes", "No")

    df = pd.DataFrame(dict(
        customerID=[f"ID-{i:05d}" for i in range(n)],
        gender=np.random.choice(["Male", "Female"], size=n),
        SeniorCitizen=senior,
        Partner=partner,
        Dependents=np.random.choice(["Yes", "No"], size=n, p=[0.30, 0.70]),
        tenure=tenure,
        PhoneService=np.random.choice(["Yes", "No"], size=n, p=[0.90, 0.10]),
        MultipleLines=np.random.choice(
            ["Yes", "No", "No phone service"], size=n, p=[0.42, 0.48, 0.10]),
        InternetService=internet,
        OnlineSecurity=np.random.choice(["Yes", "No", "No internet service"], size=n),
        TechSupport=np.random.choice(["Yes", "No", "No internet service"], size=n),
        DeviceProtection=np.random.choice(["Yes", "No", "No internet service"], size=n),
        StreamingTV=np.random.choice(["Yes", "No", "No internet service"], size=n),
        StreamingMovies=np.random.choice(["Yes", "No", "No internet service"], size=n),
        Contract=contract,
        PaperlessBilling=np.random.choice(["Yes", "No"], size=n, p=[0.59, 0.41]),
        PaymentMethod=payment,
        MonthlyCharges=monthly,
        TotalCharges=(monthly * tenure + np.random.normal(0, 50, n)).clip(0).round(2),
        Churn=churn,
    ))
    return _clean(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df["Churn_bin"] = (df["Churn"] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    # tenure buckets
    bins   = [0, 6, 12, 24, 36, 48, 60, 72]
    labels = ["0–6", "7–12", "13–24", "25–36", "37–48", "49–60", "61–72"]
    df["TenureBucket"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    # charge buckets
    cbins   = [0, 30, 50, 70, 90, 200]
    clabels = ["$0–30", "$30–50", "$50–70", "$70–90", "$90+"]
    df["ChargeBucket"] = pd.cut(df["MonthlyCharges"], bins=cbins, labels=clabels)
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train a RandomForestClassifier and return model + artefacts."""
    cat_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "TechSupport", "DeviceProtection",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    ]
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    X = df[cat_cols + num_cols].copy()
    y = df["Churn_bin"]

    le = {}
    for c in cat_cols:
        enc = LabelEncoder()
        X[c] = enc.fit_transform(X[c].astype(str))
        le[c] = enc

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)

    feat_imp = pd.DataFrame({
        "Feature":   cat_cols + num_cols,
        "Importance": clf.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return clf, feat_imp, acc, report, cm


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Churn Dashboard")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload CSV dataset",
        type=["csv"],
        help="WA_Fn-UseC_-Telco-Customer-Churn.csv",
    )

    st.markdown("---")
    st.markdown("**Filters**")

    # placeholders — filled after data loads
    contract_filter   = st.multiselect(
        "Contract type",
        ["Month-to-month", "One year", "Two year"],
        default=["Month-to-month", "One year", "Two year"],
    )
    internet_filter   = st.multiselect(
        "Internet service",
        ["Fiber optic", "DSL", "No"],
        default=["Fiber optic", "DSL", "No"],
    )
    senior_filter     = st.selectbox(
        "Senior citizen", ["All", "Senior", "Non-senior"]
    )
    tenure_range      = st.slider("Tenure range (months)", 0, 72, (0, 72))
    charge_range      = st.slider("Monthly charge range ($)", 0, 120, (0, 120))

    st.markdown("---")
    st.caption("Telco Customer Retention Analysis")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD + FILTER DATA
# ─────────────────────────────────────────────────────────────────────────────
df_raw = load_data(uploaded)

df = df_raw.copy()
if contract_filter:
    df = df[df["Contract"].isin(contract_filter)]
if internet_filter:
    df = df[df["InternetService"].isin(internet_filter)]
if senior_filter == "Senior":
    df = df[df["SeniorCitizen"] == 1]
elif senior_filter == "Non-senior":
    df = df[df["SeniorCitizen"] == 0]
df = df[df["tenure"].between(*tenure_range)]
df = df[df["MonthlyCharges"].between(*charge_range)]

churned  = df[df["Churn"] == "Yes"]
retained = df[df["Churn"] == "No"]
total    = len(df)
n_churn  = len(churned)
churn_rt = n_churn / total * 100 if total else 0

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# Telco Customer Churn Dashboard")
st.markdown(
    f"<p style='color:#888;font-size:13px;margin-top:-12px'>"
    f"WA_Fn-UseC_ Telco Customer Churn &middot; "
    f"RandomForestClassifier &middot; {total:,} records shown</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total customers",        f"{total:,}")
k2.metric("Churned",                f"{n_churn:,}",
          delta=f"{churn_rt:.1f}% churn rate", delta_color="inverse")
k3.metric("Retained",               f"{total - n_churn:,}",
          delta=f"{100 - churn_rt:.1f}% retention")
k4.metric("Avg monthly charge",     f"${df['MonthlyCharges'].mean():.2f}",
          delta=f"Churned: ${churned['MonthlyCharges'].mean():.2f}",
          delta_color="inverse")
k5.metric("Avg tenure (churned)",   f"{churned['tenure'].mean():.1f} mo",
          delta=f"Retained: {retained['tenure'].mean():.1f} mo")

st.markdown("")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview", "🔍 Deep Dive", "🤖 ML Model", "📋 Data"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    col_a, col_b, col_c = st.columns(3)

    # ── Donut ──
    with col_a:
        st.markdown("### Churn split")
        fig = go.Figure(go.Pie(
            labels=["Churned", "Retained"],
            values=[n_churn, total - n_churn],
            hole=0.68,
            marker_colors=[RED, BLUE],
            textinfo="none",
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        ))
        fig.add_annotation(
            text=f"<b>{churn_rt:.1f}%</b><br>churn",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color="#1a1917"),
            align="center",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # ── Contract bar ──
    with col_b:
        st.markdown("### Contract type")
        ct = (
            df.groupby(["Contract", "Churn"])
            .size().reset_index(name="Count")
        )
        fig = px.bar(
            ct, x="Contract", y="Count", color="Churn",
            color_discrete_map=CHURN_COLORS,
            barmode="stack",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # ── Payment bar ──
    with col_c:
        st.markdown("### Payment method")
        pm = (
            df.groupby("PaymentMethod")["Churn_bin"]
            .mean().mul(100).reset_index()
            .rename(columns={"Churn_bin": "Churn Rate %"})
            .sort_values("Churn Rate %", ascending=False)
        )
        pm["PaymentMethod"] = pm["PaymentMethod"].str.replace(
            " (automatic)", "", regex=False
        )
        colors = [
            RED if v > 35 else AMBER if v > 20 else GREEN
            for v in pm["Churn Rate %"]
        ]
        fig = go.Figure(go.Bar(
            x=pm["PaymentMethod"],
            y=pm["Churn Rate %"].round(1),
            marker_color=colors,
            text=pm["Churn Rate %"].round(1).astype(str) + "%",
            textposition="outside",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False,
                          yaxis_title="Churn rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col_d, col_e = st.columns(2)

    # ── Tenure line ──
    with col_d:
        st.markdown("### Churn rate by tenure cohort")
        tc_data = (
            df.groupby("TenureBucket", observed=True)["Churn_bin"]
            .mean().mul(100).reset_index()
            .rename(columns={"Churn_bin": "Churn Rate %"})
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tc_data["TenureBucket"].astype(str),
            y=tc_data["Churn Rate %"].round(1),
            mode="lines+markers",
            line=dict(color=RED, width=2.5),
            marker=dict(size=7, color=RED),
            fill="tozeroy",
            fillcolor="rgba(214,59,59,0.07)",
            name="Churn rate",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT, height=280,
            xaxis_title="Tenure (months)",
            yaxis_title="Churn rate (%)",
            yaxis_range=[0, 60],
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Internet service ──
    with col_e:
        st.markdown("### Churn by internet service")
        isp = (
            df.groupby(["InternetService", "Churn"])
            .size().reset_index(name="Count")
        )
        total_isp = isp.groupby("InternetService")["Count"].transform("sum")
        isp["Pct"] = (isp["Count"] / total_isp * 100).round(1)
        fig = px.bar(
            isp, x="Pct", y="InternetService",
            color="Churn", color_discrete_map=CHURN_COLORS,
            orientation="h", barmode="stack",
            text=isp["Pct"].astype(str) + "%",
        )
        fig.update_traces(textposition="inside", textfont_size=11)
        fig.update_layout(
            **PLOTLY_LAYOUT, height=280,
            xaxis_title="Percentage (%)", yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Monthly charges grouped bar ──
    st.markdown("### Monthly charges vs churn")
    ch = (
        df.groupby(["ChargeBucket", "Churn"], observed=True)
        .size().reset_index(name="Count")
    )
    fig = px.bar(
        ch, x="ChargeBucket", y="Count", color="Churn",
        color_discrete_map=CHURN_COLORS,
        barmode="group",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=260,
                      xaxis_title="Monthly charge bracket",
                      yaxis_title="Number of customers")
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    col_f, col_g = st.columns(2)

    # ── Senior citizens ──
    with col_f:
        st.markdown("### Senior citizen churn")
        sc = (
            df.groupby(["SeniorCitizen", "Churn"])
            .size().reset_index(name="Count")
        )
        sc["SeniorCitizen"] = sc["SeniorCitizen"].map(
            {0: "Non-senior", 1: "Senior"}
        )
        fig = px.bar(
            sc, x="SeniorCitizen", y="Count", color="Churn",
            color_discrete_map=CHURN_COLORS, barmode="group",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig, use_container_width=True)

    # ── Partner & dependents ──
    with col_g:
        st.markdown("### Partner / dependents churn rate")
        segments = {
            "Has partner":       df[df["Partner"] == "Yes"]["Churn_bin"].mean() * 100,
            "No partner":        df[df["Partner"] == "No"]["Churn_bin"].mean() * 100,
            "Has dependents":    df[df["Dependents"] == "Yes"]["Churn_bin"].mean() * 100,
            "No dependents":     df[df["Dependents"] == "No"]["Churn_bin"].mean() * 100,
        }
        seg_df = pd.DataFrame(list(segments.items()), columns=["Segment", "Churn Rate %"])
        colors = [RED if v > 25 else BLUE for v in seg_df["Churn Rate %"]]
        fig = go.Figure(go.Bar(
            x=seg_df["Churn Rate %"].round(1),
            y=seg_df["Segment"],
            orientation="h",
            marker_color=colors,
            text=seg_df["Churn Rate %"].round(1).astype(str) + "%",
            textposition="outside",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=280,
                          xaxis_title="Churn rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Add-on services ──
    st.markdown("### Churn rate by add-on service (with vs without)")
    addons = {
        "Online Security":    ("OnlineSecurity",    "Yes"),
        "Tech Support":       ("TechSupport",        "Yes"),
        "Device Protection":  ("DeviceProtection",   "Yes"),
        "Online Backup":      ("OnlineSecurity",    "Yes"),  # approximate
        "Streaming TV":       ("StreamingTV",        "Yes"),
        "Streaming Movies":   ("StreamingMovies",    "Yes"),
    }
    rows = []
    for name, (col, val) in addons.items():
        with_svc    = df[df[col] == val]["Churn_bin"].mean() * 100
        without_svc = df[df[col] == "No"]["Churn_bin"].mean() * 100
        rows.append({"Service": name, "With service": round(with_svc, 1),
                     "Without service": round(without_svc, 1)})
    addon_df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Without service", x=addon_df["Service"],
        y=addon_df["Without service"], marker_color=RED,
    ))
    fig.add_trace(go.Bar(
        name="With service", x=addon_df["Service"],
        y=addon_df["With service"], marker_color=GREEN,
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                      barmode="group", yaxis_title="Churn rate (%)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Scatter: tenure vs monthly charges ──
    st.markdown("### Tenure vs monthly charges coloured by churn")
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="tenure", y="MonthlyCharges", color="Churn",
        color_discrete_map=CHURN_COLORS,
        opacity=0.5, size_max=5,
        labels={"tenure": "Tenure (months)",
                "MonthlyCharges": "Monthly charges ($)"},
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(**PLOTLY_LAYOUT, height=320)
    st.plotly_chart(fig, use_container_width=True)

    # ── Risk factor table ──
    st.markdown("### Churn risk factor summary")
    risk_factors = [
        ("Electronic check payment",  45, "🔴 High"),
        ("Month-to-month contract",   43, "🔴 High"),
        ("Fiber optic internet",      42, "🔴 High"),
        ("Senior citizen",            42, "🔴 High"),
        ("No online security",        32, "🟡 Medium"),
        ("No tech support",           32, "🟡 Medium"),
        ("No dependents",             29, "🟡 Medium"),
        ("Paperless billing",         27, "🟡 Medium"),
        ("Bank transfer (auto)",      17, "🟢 Low"),
        ("1-year contract",           11, "🟢 Low"),
        ("2-year contract",            3, "🟢 Low"),
    ]
    rf_df = pd.DataFrame(risk_factors, columns=["Risk Factor", "Churn Rate (%)", "Risk Level"])
    st.dataframe(
        rf_df.style
        .bar(subset=["Churn Rate (%)"], color=["#fdf0f0", "#d63b3b"])
        .applymap(
            lambda v: "color: #d63b3b; font-weight:500" if "High" in str(v)
            else ("color: #c47d0a; font-weight:500" if "Medium" in str(v)
                  else "color: #2e8f5e; font-weight:500"),
            subset=["Risk Level"]
        ),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### RandomForest model training")
    st.info(
        "Training uses the **full unfiltered** dataset (sidebar filters do not affect the model). "
        "Click the button to train / retrain."
    )

    if st.button("Train RandomForestClassifier", type="primary"):
        with st.spinner("Training model on full dataset…"):
            clf, feat_imp, acc, report, cm = train_model(df_raw)
        st.session_state["model_trained"] = True
        st.session_state["clf"]       = clf
        st.session_state["feat_imp"]  = feat_imp
        st.session_state["acc"]       = acc
        st.session_state["report"]    = report
        st.session_state["cm"]        = cm

    if st.session_state.get("model_trained"):
        feat_imp = st.session_state["feat_imp"]
        acc      = st.session_state["acc"]
        report   = st.session_state["report"]
        cm       = st.session_state["cm"]

        # ── Model metrics ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy",  f"{acc*100:.1f}%")
        m2.metric("Precision (churn)", f"{report['1']['precision']*100:.1f}%")
        m3.metric("Recall (churn)",    f"{report['1']['recall']*100:.1f}%")
        m4.metric("F1-score (churn)",  f"{report['1']['f1-score']*100:.1f}%")

        st.markdown("---")
        col_h, col_i = st.columns(2)

        # ── Feature importance ──
        with col_h:
            st.markdown("### Feature importance (top 15)")
            top = feat_imp.head(15).sort_values("Importance")
            fig = go.Figure(go.Bar(
                x=top["Importance"],
                y=top["Feature"],
                orientation="h",
                marker=dict(
                    color=top["Importance"],
                    colorscale=[[0, BLUE], [1, RED]],
                    showscale=False,
                ),
                text=(top["Importance"] * 100).round(1).astype(str) + "%",
                textposition="outside",
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT, height=420,
                xaxis_title="Importance score",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Confusion matrix ──
        with col_i:
            st.markdown("### Confusion matrix")
            labels = ["Retained (0)", "Churned (1)"]
            fig = go.Figure(go.Heatmap(
                z=cm,
                x=labels, y=labels,
                colorscale=[[0, "#eef5fb"], [1, BLUE]],
                text=cm,
                texttemplate="%{text:,}",
                textfont=dict(size=16),
                showscale=False,
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT, height=420,
                xaxis_title="Predicted",
                yaxis_title="Actual",
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Full classification report ──
        st.markdown("### Classification report")
        report_df = pd.DataFrame(report).transpose().round(3)
        report_df = report_df[report_df.index.isin(["0", "1", "macro avg", "weighted avg"])]
        report_df.index = ["Retained", "Churned", "Macro avg", "Weighted avg"]
        st.dataframe(
            report_df.style.format("{:.3f}").background_gradient(
                cmap="Blues", subset=["precision", "recall", "f1-score"]
            ),
            use_container_width=True,
        )

        # ── Predict for new customer ──
        st.markdown("---")
        st.markdown("### Predict churn for a new customer")
        p1, p2, p3 = st.columns(3)
        with p1:
            p_tenure   = st.slider("Tenure (months)", 1, 72, 12)
            p_monthly  = st.slider("Monthly charges ($)", 18, 120, 65)
            p_contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with p2:
            p_internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            p_payment  = st.selectbox("Payment method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            p_senior   = st.selectbox("Senior citizen", ["No", "Yes"])
        with p3:
            p_partner  = st.selectbox("Partner", ["Yes", "No"])
            p_dep      = st.selectbox("Dependents", ["Yes", "No"])
            p_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])

        if st.button("Predict churn probability"):
            clf   = st.session_state["clf"]
            # build a single-row dataframe matching training columns
            row = {
                "gender": "Male", "SeniorCitizen": int(p_senior == "Yes"),
                "Partner": p_partner, "Dependents": p_dep,
                "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": p_internet, "OnlineSecurity": p_security,
                "TechSupport": "No", "DeviceProtection": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": p_contract, "PaperlessBilling": "Yes",
                "PaymentMethod": p_payment,
                "tenure": p_tenure,
                "MonthlyCharges": p_monthly,
                "TotalCharges": p_monthly * p_tenure,
            }
            X_new = pd.DataFrame([row])
            cat_cols = [
                "gender", "SeniorCitizen", "Partner", "Dependents",
                "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "TechSupport", "DeviceProtection",
                "StreamingTV", "StreamingMovies", "Contract",
                "PaperlessBilling", "PaymentMethod",
            ]
            for c in cat_cols:
                le = LabelEncoder()
                le.fit(df_raw[c].astype(str).unique())
                try:
                    X_new[c] = le.transform(X_new[c].astype(str))
                except ValueError:
                    X_new[c] = 0

            proba = clf.predict_proba(X_new)[0][1]
            risk  = "🔴 High" if proba > 0.6 else "🟡 Medium" if proba > 0.35 else "🟢 Low"
            st.markdown(
                f"<div style='padding:16px;background:#f8f7f3;border:1px solid #e0deda;"
                f"border-radius:10px;font-size:16px'>"
                f"Churn probability: <b style='font-size:24px'>{proba*100:.1f}%</b> &nbsp; {risk}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div style='padding:24px;background:#f8f7f3;border-radius:10px;"
            "color:#7a7870;text-align:center'>"
            "Click the button above to train the model.</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Dataset preview")
    show_cols = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "Contract", "InternetService", "PaymentMethod",
        "MonthlyCharges", "TotalCharges", "Churn",
    ]
    st.dataframe(
        df[show_cols].head(200),
        use_container_width=True,
        height=340,
    )

    st.markdown("---")
    col_j, col_k = st.columns(2)

    with col_j:
        st.markdown("### Descriptive statistics (numeric)")
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        st.dataframe(
            df[num_cols].describe().round(2),
            use_container_width=True,
        )

    with col_k:
        st.markdown("### Class balance")
        cb = df["Churn"].value_counts().reset_index()
        cb.columns = ["Churn", "Count"]
        cb["Percentage"] = (cb["Count"] / cb["Count"].sum() * 100).round(1)
        st.dataframe(cb, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.download_button(
        label="⬇ Download filtered data as CSV",
        data=df[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="churn_filtered.csv",
        mime="text/csv",
    )
