
import os
import json
import joblib

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Credit Default Risk Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
/* ══════════════════════════════════════════════════
   DARK BASE
══════════════════════════════════════════════════ */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
    background-color: #0f1117 !important;
    color: #e2e8f0 !important;
}
[data-testid="stAppViewContainer"] > .main { background-color: #0f1117; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0d14 !important;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] *     { color: #cbd5e0 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3    { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stMetric label {
    color: #718096 !important; font-size: 0.78rem !important;
}
[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
    color: #63b3ed !important; font-size: 1.1rem !important;
}
[data-testid="stSidebar"] hr { border-color: #1e2535 !important; }
[data-testid="stSidebar"] .stMarkdown table td,
[data-testid="stSidebar"] .stMarkdown table th {
    color: #cbd5e0 !important;
    border-color: #2d3748 !important;
    background: transparent !important;
}

/* ── All text ── */
h1, h2, h3, h4, h5, h6 { color: #f0f4f8 !important; }
p, li, span, label, div { color: #cbd5e0; }
.stMarkdown,
.stCaption,
[data-testid="stCaptionContainer"] { color: #94a3b8 !important; }

/* ── Metric widget ── */
[data-testid="stMetric"] {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
[data-testid="stMetricValue"] { color: #63b3ed !important; }
[data-testid="stMetricDelta"] { color: #68d391 !important; }

/* ── Native inputs ── */
[data-baseweb="select"] > div,
[data-baseweb="input"]  > div,
[data-baseweb="textarea"] {
    background-color: #1e2535 !important;
    border-color: #2d3748 !important;
    color: #e2e8f0 !important;
}
[data-baseweb="select"] span,
[data-baseweb="input"]  input { color: #e2e8f0 !important; }

/* Dropdown popup */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"] {
    background-color: #1e2535 !important;
    border: 1px solid #2d3748 !important;
}
[data-baseweb="menu"] li       { color: #cbd5e0 !important; }
[data-baseweb="menu"] li:hover { background: #2d3748 !important; }

/* Slider */
[data-testid="stSlider"] [role="slider"] { background: #4299e1 !important; }

/* Checkbox */
[data-baseweb="checkbox"] [aria-checked] {
    background-color: #4299e1 !important;
    border-color: #4299e1 !important;
}

/* Toggle */
[data-testid="stToggle"] [data-checked="true"] > div { background: #4299e1 !important; }

/* ── Divider ── */
hr { border-color: #2d3748 !important; }

/* ── DataTable ── */
[data-testid="stDataFrame"] iframe,
[data-testid="stDataFrame"] * { background: #1a1f2e !important; color: #e2e8f0 !important; }

/* ── Alert / info / error ── */
[data-testid="stAlert"] {
    background: #1e2535 !important;
    border-left-color: #4299e1 !important;
    color: #cbd5e0 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #1a1f2e !important;
    border: 1px solid #2d3748 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * { color: #94a3b8 !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] * { color: #63b3ed !important; }

/* ══════════════════════════════════════════════════
   SECTION TITLE
══════════════════════════════════════════════════ */
.section-title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4299e1 !important;
    margin-bottom: 0.8rem;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #2d3748;
}

/* ══════════════════════════════════════════════════
   PREDICT BUTTON
══════════════════════════════════════════════════ */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #1a4a7a, #2b6cb0);
    color: #f0f4f8 !important;
    border: 1px solid #2b6cb0;
    border-radius: 10px;
    padding: 0.8rem 2rem;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    cursor: pointer;
    transition: all 0.2s;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #2b6cb0, #4299e1);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(66,153,225,0.5);
}

/* ══════════════════════════════════════════════════
   TABS
══════════════════════════════════════════════════ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #1a1f2e;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #2d3748;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    color: #94a3b8 !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #2d3748 !important;
    color: #e2e8f0 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.4);
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] { background: transparent; }

/* ══════════════════════════════════════════════════
   CAPTION & SMALL TEXT
══════════════════════════════════════════════════ */
.stCaption,
[data-testid="stCaptionContainer"] small { color: #718096 !important; }
</style>
""",
    unsafe_allow_html=True,
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

NAME_CONTRACT_TYPE_OPTS = ["Cash loans", "Revolving loans"]
CODE_GENDER_OPTS        = ["M", "F", "XNA"]
FLAG_OPTS               = ["Y", "N"]

NAME_INCOME_TYPE_OPTS = [
    "Working", "Commercial associate", "Pensioner", "State servant",
    "Unemployed", "Student", "Businessman", "Maternity leave",
]
NAME_EDUCATION_TYPE_OPTS = [
    "Secondary / secondary special", "Higher education",
    "Incomplete higher", "Lower secondary", "Academic degree",
]
NAME_FAMILY_STATUS_OPTS = [
    "Married", "Single / not married", "Civil marriage", "Separated", "Widow",
]
NAME_HOUSING_TYPE_OPTS = [
    "House / apartment", "With parents", "Municipal apartment",
    "Rented apartment", "Office apartment", "Co-op apartment",
]
OCCUPATION_TYPE_OPTS = [
    "Not specified", "Laborers", "Sales staff", "Core staff", "Managers",
    "Drivers", "High skill tech staff", "Accountants", "Medicine staff",
    "Security staff", "Cooking staff", "Cleaning staff",
    "Private service staff", "Low-skill Laborers", "Waiters/barmen staff",
    "Secretaries", "Realty agents", "HR staff", "IT staff",
]
ORGANIZATION_TYPE_OPTS = [
    "XNA", "Business Entity Type 3", "School", "Government", "Religion",
    "Other", "Medicine", "Business Entity Type 2", "Self-employed",
    "Transport: type 2", "Construction", "Housing", "Kindergarten",
    "Trade: type 7", "Industry: type 11", "Military", "Services",
    "Security Ministries", "Transport: type 4", "Industry: type 1",
    "Emergency", "Security", "Trade: type 2", "University",
    "Transport: type 3", "Police", "Business Entity Type 1",
    "Postal", "Industry: type 4", "Agriculture", "Restaurant",
    "Culture", "Hotel", "Industry: type 7", "Trade: type 3",
    "Industry: type 3", "Bank", "Industry: type 9", "Insurance",
    "Trade: type 6", "Industry: type 2", "Transport: type 1",
    "Industry: type 12", "Mobile", "Trade: type 1", "Industry: type 5",
    "Industry: type 10", "Legal Services", "Advertising",
    "Trade: type 4", "Electricity", "Trade: type 5",
    "Industry: type 13", "Telecom", "Industry: type 6",
]

WEEKDAY_OPTS = [
    "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY",
]

@st.cache_resource(show_spinner="⏳  Loading model artifacts…")
def load_artifacts():
    if not os.path.isdir(ARTIFACTS_DIR):
        st.error(
            f"Artifacts folder not found at `{ARTIFACTS_DIR}`. "
            "Please run all cells of `fraud_detection_professional.ipynb` first."
        )
        st.stop()

    model     = joblib.load(os.path.join(ARTIFACTS_DIR, "model.joblib"))
    scaler    = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    imputer   = joblib.load(os.path.join(ARTIFACTS_DIR, "imputer.joblib"))
    le_dict   = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoders.joblib"))
    feat_cols = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.joblib"))

    with open(os.path.join(ARTIFACTS_DIR, "model_metadata.json"))       as fh:
        metadata = json.load(fh)
    with open(os.path.join(ARTIFACTS_DIR, "preprocessing_config.json")) as fh:
        config   = json.load(fh)

    return model, scaler, imputer, le_dict, feat_cols, metadata, config


def rule_based_assessment(raw: dict, prev_app: dict | None) -> dict:
    rules_fired: list[dict] = []
    score = 0.0

    income        = float(raw.get("AMT_INCOME_TOTAL") or 1)
    credit        = float(raw.get("AMT_CREDIT")       or 0)
    annuity       = float(raw.get("AMT_ANNUITY")      or 0)
    goods         = float(raw.get("AMT_GOODS_PRICE")  or 1)
    days_birth    = float(raw.get("DAYS_BIRTH",  -10000) or -10000)
    days_employed = float(raw.get("DAYS_EMPLOYED",    0) or 0)

    age                  = -days_birth / 365
    credit_income_ratio  = credit  / max(income, 1)
    annuity_income_ratio = annuity / max(income, 1)
    credit_goods_ratio   = credit  / max(goods,  1)
    employed_anom        = (days_employed == 365243)

    ext_vals = [
        float(raw[k]) for k in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
        if raw.get(k) not in (None, "", 0)
    ]
    ext_mean = (sum(ext_vals) / len(ext_vals)) if ext_vals else None

    doc_cols      = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]
    docs_provided = sum(int(raw.get(c, 0) or 0) for c in doc_cols)
    income_type   = str(raw.get("NAME_INCOME_TYPE", "") or "")

    if credit_income_ratio > 8:
        score += 0.45
        rules_fired.append({
            "sev": "HIGH",
            "msg": f"Credit ({credit:,.0f}) is {credit_income_ratio:.1f}× annual income — extreme over-leverage (threshold: 8×)",
        })
    elif credit_income_ratio > 5:
        score += 0.20
        rules_fired.append({
            "sev": "MEDIUM",
            "msg": f"Credit-to-income ratio {credit_income_ratio:.1f}× exceeds 5× — high leverage",
        })

    if annuity_income_ratio > 0.50:
        score += 0.40
        rules_fired.append({
            "sev": "HIGH",
            "msg": f"Annuity is {annuity_income_ratio*100:.0f}% of income (>50%) — repayment likely unaffordable",
        })
    elif annuity_income_ratio > 0.35:
        score += 0.18
        rules_fired.append({
            "sev": "MEDIUM",
            "msg": f"Annuity is {annuity_income_ratio*100:.0f}% of income (>35%) — stressed repayment capacity",
        })

    if ext_mean is None:
        score += 0.15
        rules_fired.append({
            "sev": "MEDIUM",
            "msg": "No external credit scores available — creditworthiness cannot be independently verified",
        })
    elif ext_mean < 0.10:
        score += 0.50
        rules_fired.append({
            "sev": "HIGH",
            "msg": f"External score mean {ext_mean:.3f} is critically low (<0.10) — very poor credit history",
        })
    elif ext_mean < 0.20:
        score += 0.28
        rules_fired.append({
            "sev": "HIGH",
            "msg": f"External score mean {ext_mean:.3f} is below 0.20 — below-average credit history",
        })
    elif ext_mean < 0.30:
        score += 0.12
        rules_fired.append({
            "sev": "MEDIUM",
            "msg": f"External score mean {ext_mean:.3f} is marginal (<0.30)",
        })

    if age < 22:
        score += 0.22
        rules_fired.append({
            "sev": "MEDIUM",
            "msg": f"Applicant is {age:.0f} years old — very young, minimal credit history expected",
        })
    elif age < 25:
        score += 0.10
        rules_fired.append({
            "sev": "LOW",
            "msg": f"Applicant is {age:.0f} years old — limited credit history",
        })

    if employed_anom:
        score += 0.30
        rules_fired.append({
            "sev": "HIGH",
            "msg": "Employment duration is anomalous (365243 days) — likely retired or unemployed in source data",
        })
    if income_type in ("Unemployed", "Student", "Maternity leave"):
        score += 0.28
        rules_fired.append({
            "sev": "HIGH",
            "msg": f"Income type '{income_type}' indicates no stable employment income",
        })

    if docs_provided == 0:
        score += 0.20
        rules_fired.append({
            "sev": "MEDIUM",
            "msg": "Zero supporting documents provided — identity and income cannot be verified",
        })

    if credit_goods_ratio > 1.35:
        score += 0.15
        rules_fired.append({
            "sev": "MEDIUM",
            "msg": f"Credit ({credit:,.0f}) exceeds goods price ({goods:,.0f}) by {(credit_goods_ratio-1)*100:.0f}% — possible cash-out or over-valuation",
        })

    if prev_app:
        approval_rate = prev_app.get("PREV_APPROVAL_RATE")
        if approval_rate is not None:
            if approval_rate < 0.25:
                score += 0.30
                rules_fired.append({
                    "sev": "HIGH",
                    "msg": f"Previous loan approval rate is only {approval_rate*100:.0f}% — strong history of rejections",
                })
            elif approval_rate < 0.50:
                score += 0.12
                rules_fired.append({
                    "sev": "MEDIUM",
                    "msg": f"Previous loan approval rate is {approval_rate*100:.0f}% — moderate rejection history",
                })

    score = min(score, 1.0)

    if   score >= 0.60: rule_label, rule_color, rule_emoji, rule_pred = "VERY HIGH RISK", "#E53E3E", "🔴", 1
    elif score >= 0.40: rule_label, rule_color, rule_emoji, rule_pred = "HIGH RISK",      "#DD6B20", "🟠", 1
    elif score >= 0.20: rule_label, rule_color, rule_emoji, rule_pred = "MODERATE RISK",  "#D69E2E", "🟡", 0
    else:               rule_label, rule_color, rule_emoji, rule_pred = "LOW RISK",       "#38A169", "🟢", 0

    return {
        "score":       round(score, 4),
        "prediction":  rule_pred,
        "risk_label":  rule_label,
        "risk_color":  rule_color,
        "risk_emoji":  rule_emoji,
        "rules_fired": rules_fired,
    }


def predict_fraud(raw_input: dict, prev_app: dict | None = None) -> dict:
    model, scaler, imputer, le_dict, feat_cols, metadata, config = load_artifacts()
    threshold = metadata["optimal_threshold"]

    row = {k: v for k, v in raw_input.items() if k not in ("SK_ID_CURR", "TARGET")}
    if prev_app:
        row.update(prev_app)
    sample = pd.DataFrame([row])

    if "DAYS_BIRTH" in sample.columns:
        sample["AGE_YEARS"] = -sample["DAYS_BIRTH"] / 365

    if "DAYS_EMPLOYED" in sample.columns:
        sample["DAYS_EMPLOYED_ANOM"] = (sample["DAYS_EMPLOYED"] == 365243).astype(int)
        sample["DAYS_EMPLOYED"]      = sample["DAYS_EMPLOYED"].replace(365243, np.nan)
        sample["EMPLOYED_YEARS"]     = -sample["DAYS_EMPLOYED"] / 365

    for new_col, num, den in [
        ("CREDIT_INCOME_RATIO",   "AMT_CREDIT",  "AMT_INCOME_TOTAL"),
        ("ANNUITY_INCOME_RATIO",  "AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        ("CREDIT_TERM",           "AMT_ANNUITY", "AMT_CREDIT"),
        ("CREDIT_TO_GOODS_RATIO", "AMT_CREDIT",  "AMT_GOODS_PRICE"),
    ]:
        if num in sample.columns and den in sample.columns:
            sample[new_col] = sample[num] / (sample[den] + 1)

    ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
           if c in sample.columns]
    if ext:
        sample["EXT_SOURCE_MEAN"] = sample[ext].mean(axis=1)
        sample["EXT_SOURCE_MIN"]  = sample[ext].min(axis=1)
        sample["EXT_SOURCE_MAX"]  = sample[ext].max(axis=1)

    flag_cols = [c for c in sample.columns if c.startswith("FLAG_DOCUMENT")]
    if flag_cols:
        sample["DOCUMENTS_PROVIDED"] = sample[flag_cols].sum(axis=1)

    drop_all = (
        config["dropped_high_missing_cols"]
        + config["dropped_raw_cols"]
        + config["dropped_id_cols"]
    )
    sample.drop(columns=[c for c in drop_all if c in sample.columns], inplace=True)

    for col, le in le_dict.items():
        if col in sample.columns:
            sample[col] = sample[col].fillna("Unknown")
            try:
                sample[col] = le.transform(sample[col])
            except ValueError:
                sample[col] = 0

    ohe_present = [c for c in config["ohe_categorical_cols"] if c in sample.columns]
    if ohe_present:
        sample = pd.get_dummies(sample, columns=ohe_present,
                                drop_first=True, dummy_na=False)
        bc = sample.select_dtypes(include="bool").columns
        sample[bc] = sample[bc].astype(int)

    sample = sample.reindex(columns=feat_cols, fill_value=0)

    imputed = imputer.transform(sample)
    scaled  = scaler.transform(imputed)
    prob    = float(model.predict_proba(scaled)[0, 1])
    model_pred = int(prob >= threshold)

    if   prob >= 0.70: ml_label, ml_color, ml_emoji = "VERY HIGH RISK", "#E53E3E", "🔴"
    elif prob >= 0.50: ml_label, ml_color, ml_emoji = "HIGH RISK",      "#DD6B20", "🟠"
    elif prob >= 0.30: ml_label, ml_color, ml_emoji = "MODERATE RISK",  "#D69E2E", "🟡"
    else:              ml_label, ml_color, ml_emoji = "LOW RISK",        "#38A169", "🟢"

    rule = rule_based_assessment(row, prev_app)

    if rule["prediction"] == 1 and model_pred == 0:
        final_pred   = 1
        driver       = "rule"
        risk_label   = rule["risk_label"]
        risk_color   = rule["risk_color"]
        risk_emoji   = rule["risk_emoji"]
    elif rule["prediction"] == 1 and model_pred == 1:
        final_pred   = 1
        driver       = "both"
        if rule["score"] >= prob:
            risk_label, risk_color, risk_emoji = rule["risk_label"], rule["risk_color"], rule["risk_emoji"]
        else:
            risk_label, risk_color, risk_emoji = ml_label, ml_color, ml_emoji
    elif rule["prediction"] == 0 and model_pred == 1:
        final_pred   = 1
        driver       = "model"
        risk_label, risk_color, risk_emoji = ml_label, ml_color, ml_emoji
    else:
        final_pred   = 0
        driver       = "none"
        risk_label, risk_color, risk_emoji = ml_label, ml_color, ml_emoji

    return {
        "probability":   round(prob,      4),
        "prediction":    final_pred,
        "risk_label":    risk_label,
        "risk_color":    risk_color,
        "risk_emoji":    risk_emoji,
        "threshold":     round(threshold, 4),
        "ml_prob":       round(prob,      4),
        "ml_label":      ml_label,
        "ml_color":      ml_color,
        "ml_emoji":      ml_emoji,
        "ml_pred":       model_pred,
        "rule":          rule,
        "driver":        driver,
        "top_features":  dict(list(metadata["top_20_feature_importances"].items())[:10]),
        "all_features":  metadata["top_20_feature_importances"],
        "all_models":    metadata.get("all_model_comparison", {}),
        "dataset_info":  metadata["dataset_info"],
        "perf":          metadata["performance_metrics"],
    }


def render_sidebar(metadata: dict) -> None:
    with st.sidebar:
        st.markdown(
            "<h1 style='margin-bottom:0;font-size:1.5rem;'>🏦 Credit Risk<br>Analyzer</h1>",
            unsafe_allow_html=True,
        )
        st.caption("Home Credit Default · Random Forest + SMOTE")
        st.divider()

        st.markdown("**📊 Model Performance**")
        p = metadata["performance_metrics"]
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC",       f'{p["test_roc_auc"]:.3f}')
        c2.metric("Avg Precision", f'{p["test_avg_precision"]:.3f}')
        c1.metric("F1 Score",      f'{p["test_f1_score"]:.3f}')
        c2.metric("MCC",           f'{p["test_mcc"]:.3f}')

        st.divider()
        st.markdown("**🔧 Model Details**")
        d = metadata["dataset_info"]
        st.markdown(
            f"""
| | |
|---|---|
| Algorithm | Random Forest |
| Threshold | `{metadata['optimal_threshold']:.3f}` |
| Features  | `{d['n_features']}` |
| Train set | `{d['train_samples_raw']:,}` rows |
| SMOTE set | `{d['train_samples_smote']:,}` rows |
| Test set  | `{d['test_samples']:,}` rows |
| Default rate | `{d['fraud_rate_pct']:.2f}%` |
"""
        )

        st.divider()
        st.markdown("**ℹ️ About**")
        st.info(
            "Fill in the loan application form and click **Analyze Risk** "
            "to get the model's default probability, risk tier, and "
            "the features that drove the decision."
        )

        st.divider()
        st.caption("Built from `fraud_detection_professional.ipynb`")


def render_input_form() -> tuple[dict, dict | None]:
    tabs = st.tabs(
        ["👤 Personal", "💰 Financial", "🏢 Employment",
         "📊 Credit Scores", "📋 Previous Loans", "📄 Documents & Flags"]
    )

    raw: dict  = {}
    prev: dict = {}

    with tabs[0]:
        st.markdown('<p class="section-title">Personal & Demographic Information</p>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        age_years = c1.slider("Age (years)", 18, 70, 35,
                               help="Applicant's age in years")
        raw["DAYS_BIRTH"] = -int(age_years * 365)

        raw["CODE_GENDER"] = c2.selectbox(
            "Gender", CODE_GENDER_OPTS,
            help="M = Male, F = Female, XNA = Not disclosed")

        raw["CNT_CHILDREN"] = c3.number_input(
            "Number of Children", 0, 19, 0, step=1)

        c1, c2, c3 = st.columns(3)
        raw["CNT_FAM_MEMBERS"] = c1.number_input(
            "Family Members", 1.0, 20.0, float(raw["CNT_CHILDREN"] + 1),
            step=1.0)
        raw["NAME_FAMILY_STATUS"] = c2.selectbox(
            "Family Status", NAME_FAMILY_STATUS_OPTS)
        raw["NAME_EDUCATION_TYPE"] = c3.selectbox(
            "Education Level", NAME_EDUCATION_TYPE_OPTS)

        c1, c2, c3 = st.columns(3)
        raw["FLAG_OWN_CAR"]    = c1.selectbox("Owns a Car?",    FLAG_OPTS)
        raw["FLAG_OWN_REALTY"] = c2.selectbox("Owns Property?", FLAG_OPTS)
        raw["NAME_HOUSING_TYPE"] = c3.selectbox(
            "Housing Type", NAME_HOUSING_TYPE_OPTS)

        c1, c2 = st.columns(2)
        raw["REGION_POPULATION_RELATIVE"] = c1.number_input(
            "Region Population Relative", 0.0, 0.1, 0.02, step=0.001,
            format="%.4f",
            help="Normalised population of the applicant's region (0–0.072 typical range)")
        raw["WEEKDAY_APPR_PROCESS_START"] = c2.selectbox(
            "Application Weekday", WEEKDAY_OPTS, index=0)

        c1, c2 = st.columns(2)
        raw["REG_CITY_NOT_WORK_CITY"]   = int(c1.checkbox(
            "Registration city ≠ work city"))
        raw["LIVE_CITY_NOT_WORK_CITY"]  = int(c2.checkbox(
            "Living city ≠ work city"))

    with tabs[1]:
        st.markdown('<p class="section-title">Loan & Financial Details</p>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        raw["NAME_CONTRACT_TYPE"] = c1.selectbox(
            "Contract Type", NAME_CONTRACT_TYPE_OPTS)
        raw["NAME_INCOME_TYPE"] = c2.selectbox(
            "Income Type", NAME_INCOME_TYPE_OPTS)

        c1, c2 = st.columns(2)
        raw["AMT_INCOME_TOTAL"] = c1.number_input(
            "Annual Income (₽)", 25_000.0, 20_000_000.0, 180_000.0,
            step=5_000.0, format="%.0f",
            help="Total annual income of the applicant")
        raw["AMT_CREDIT"] = c2.number_input(
            "Loan Amount (₽)", 45_000.0, 4_050_000.0, 500_000.0,
            step=10_000.0, format="%.0f",
            help="Credit amount of the loan")

        c1, c2 = st.columns(2)
        raw["AMT_ANNUITY"] = c1.number_input(
            "Loan Annuity (₽/month)", 1_000.0, 250_000.0, 25_000.0,
            step=500.0, format="%.0f",
            help="Monthly payment amount")
        raw["AMT_GOODS_PRICE"] = c2.number_input(
            "Goods Price (₽)", 40_500.0, 4_050_000.0, 450_000.0,
            step=10_000.0, format="%.0f",
            help="Price of the goods for consumer loans; 0 for cash loans")

        credit_income = raw["AMT_CREDIT"] / (raw["AMT_INCOME_TOTAL"] + 1)
        annuity_income = raw["AMT_ANNUITY"] / (raw["AMT_INCOME_TOTAL"] + 1)
        credit_term = (raw["AMT_CREDIT"] / raw["AMT_ANNUITY"]) / 12 \
            if raw["AMT_ANNUITY"] > 0 else 0

        st.markdown("**Auto-computed ratios (for reference)**")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Credit / Income",     f"{credit_income:.2f}x",
                   delta=None, help="Lower is safer")
        mc2.metric("Annuity / Income",    f"{annuity_income:.3f}",
                   delta=None, help="Monthly burden as fraction of income")
        mc3.metric("Loan Term",           f"{credit_term:.1f} yrs",
                   help="Implied loan term in years")

    with tabs[2]:
        st.markdown('<p class="section-title">Employment & Organisation</p>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        employed_years = c1.slider(
            "Years Employed", 0, 50, 5,
            help="0 = currently unemployed / pensioner")
        if employed_years == 0:
            raw["DAYS_EMPLOYED"] = 365243
        else:
            raw["DAYS_EMPLOYED"] = -int(employed_years * 365)

        raw["OCCUPATION_TYPE"] = c2.selectbox(
            "Occupation", OCCUPATION_TYPE_OPTS, index=0)

        c1, c2 = st.columns(2)
        raw["ORGANIZATION_TYPE"] = c1.selectbox(
            "Organisation Type", ORGANIZATION_TYPE_OPTS, index=0,
            help="Type of employer organisation")

        raw["DAYS_REGISTRATION"] = -int(
            c2.slider("Years Since Registration", 0, 40, 10) * 365
        )

        c1, c2 = st.columns(2)
        raw["DAYS_ID_PUBLISH"] = -int(
            c1.slider("Years Since ID Issued", 0, 25, 5) * 365)
        raw["HOUR_APPR_PROCESS_START"] = c2.slider(
            "Application Hour", 0, 23, 10,
            help="Hour of the day the application was submitted")

        c1, c2 = st.columns(2)
        raw["FLAG_EMP_PHONE"]   = int(c1.checkbox("Has employment phone"))
        raw["FLAG_WORK_PHONE"]  = int(c2.checkbox("Has work phone"))

    with tabs[3]:
        st.markdown('<p class="section-title">External Credit Scores</p>',
                    unsafe_allow_html=True)
        st.info(
            "EXT_SOURCE scores are external credit bureau scores normalised to **[0, 1]**. "
            "Higher values indicate **better creditworthiness**. "
            "Leave at `0.0` if unknown — the imputer will fill the median."
        )

        c1, c2, c3 = st.columns(3)
        raw["EXT_SOURCE_1"] = c1.slider(
            "EXT_SOURCE_1", 0.0, 1.0, 0.50, step=0.01,
            help="External score from source 1 (normalised credit bureau score)")
        raw["EXT_SOURCE_2"] = c2.slider(
            "EXT_SOURCE_2", 0.0, 1.0, 0.50, step=0.01,
            help="External score from source 2")
        raw["EXT_SOURCE_3"] = c3.slider(
            "EXT_SOURCE_3", 0.0, 1.0, 0.50, step=0.01,
            help="External score from source 3")

        avg_ext = (raw["EXT_SOURCE_1"] + raw["EXT_SOURCE_2"] + raw["EXT_SOURCE_3"]) / 3
        ext_color = "#38A169" if avg_ext >= 0.6 else "#D69E2E" if avg_ext >= 0.4 else "#E53E3E"
        st.markdown(
            f'<p style="color:{ext_color};font-weight:700;font-size:1rem;">'
            f'Average external score: {avg_ext:.3f} — '
            f'{"Good ✅" if avg_ext >= 0.6 else "Moderate ⚠️" if avg_ext >= 0.4 else "Poor ❌"}'
            f"</p>",
            unsafe_allow_html=True,
        )

    with tabs[4]:
        st.markdown('<p class="section-title">Previous Loan Applications (optional)</p>',
                    unsafe_allow_html=True)
        has_prev = st.toggle(
            "Include previous application history",
            value=False,
            help="If disabled, these features are filled with the training median.")

        if has_prev:
            c1, c2, c3 = st.columns(3)
            total = c1.number_input("Total prev. applications", 0, 100, 2, step=1)
            approved = c2.number_input("Approved applications",   0, total, 1, step=1)
            refused  = c3.number_input("Refused applications",    0, total, 0, step=1)

            c1, c2 = st.columns(2)
            amt_mean = c1.number_input("Mean credit amount (prev.)",
                                       0.0, 4_000_000.0, 400_000.0,
                                       step=10_000.0, format="%.0f")
            amt_max  = c2.number_input("Max credit amount (prev.)",
                                       0.0, 4_000_000.0, 600_000.0,
                                       step=10_000.0, format="%.0f")

            c1, c2 = st.columns(2)
            ann_mean = c1.number_input("Mean annuity (prev.)",
                                       0.0, 200_000.0, 20_000.0,
                                       step=1_000.0, format="%.0f")
            days_dec = c2.slider("Avg days since prev. decision", -3000, 0, -500)

            prev = {
                "PREV_COUNT":               int(total),
                "PREV_APPROVED_COUNT":      int(approved),
                "PREV_REFUSED_COUNT":       int(refused),
                "PREV_AMT_CREDIT_MEAN":     float(amt_mean),
                "PREV_AMT_CREDIT_MAX":      float(amt_max),
                "PREV_AMT_ANNUITY_MEAN":    float(ann_mean),
                "PREV_DAYS_DECISION_MEAN":  float(days_dec),
                "PREV_APPROVAL_RATE":       (float(approved) / total) if total > 0 else np.nan,
            }
        else:
            prev = None

    with tabs[5]:
        st.markdown('<p class="section-title">Document Flags</p>',
                    unsafe_allow_html=True)

        st.info(
            "**Why are they just numbered?**  \n"
            "The Home Credit dataset anonymizes all document types for privacy reasons. "
            "The bank knows internally what Document 3, Document 6, etc. refer to "
            "(e.g. passport, birth certificate, employment letter, tax ID, etc.) "
            "but this mapping was **not disclosed** in the public dataset.  \n\n"
            "✅ **What matters to the model:** not which specific document, but the "
            "**total count** of documents provided. More documents = more verified applicant "
            "= generally lower default risk.  \n\n"
            "👇 **Tick every document the applicant has submitted.**"
        )

        doc_cols = [
            "FLAG_DOCUMENT_2",  "FLAG_DOCUMENT_3",  "FLAG_DOCUMENT_4",
            "FLAG_DOCUMENT_5",  "FLAG_DOCUMENT_6",  "FLAG_DOCUMENT_7",
            "FLAG_DOCUMENT_8",  "FLAG_DOCUMENT_9",  "FLAG_DOCUMENT_10",
            "FLAG_DOCUMENT_11", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13",
            "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16",
            "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19",
            "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21",
        ]

        doc_labels = {
            "FLAG_DOCUMENT_2" : "Doc 2  — Identity document (type A)",
            "FLAG_DOCUMENT_3" : "Doc 3  — Identity / registration (most common)",
            "FLAG_DOCUMENT_4" : "Doc 4  — Address proof",
            "FLAG_DOCUMENT_5" : "Doc 5  — Income / employment proof",
            "FLAG_DOCUMENT_6" : "Doc 6  — Social / pension document",
            "FLAG_DOCUMENT_7" : "Doc 7  — Supporting document 7",
            "FLAG_DOCUMENT_8" : "Doc 8  — Supporting document 8",
            "FLAG_DOCUMENT_9" : "Doc 9  — Supporting document 9",
            "FLAG_DOCUMENT_10": "Doc 10 — Supporting document 10",
            "FLAG_DOCUMENT_11": "Doc 11 — Supporting document 11",
            "FLAG_DOCUMENT_12": "Doc 12 — Supporting document 12",
            "FLAG_DOCUMENT_13": "Doc 13 — Supporting document 13",
            "FLAG_DOCUMENT_14": "Doc 14 — Supporting document 14",
            "FLAG_DOCUMENT_15": "Doc 15 — Supporting document 15",
            "FLAG_DOCUMENT_16": "Doc 16 — Supporting document 16",
            "FLAG_DOCUMENT_17": "Doc 17 — Supporting document 17",
            "FLAG_DOCUMENT_18": "Doc 18 — Supporting document 18",
            "FLAG_DOCUMENT_19": "Doc 19 — Supporting document 19",
            "FLAG_DOCUMENT_20": "Doc 20 — Supporting document 20",
            "FLAG_DOCUMENT_21": "Doc 21 — Supporting document 21",
        }

        g_cols = st.columns(4)
        for i, col in enumerate(doc_cols):
            default_on = (col == "FLAG_DOCUMENT_3")
            raw[col] = int(
                g_cols[i % 4].checkbox(doc_labels[col], value=default_on, key=col)
            )

        doc_count = sum(raw[c] for c in doc_cols)
        count_color = "#68d391" if doc_count >= 3 else "#f6ad55" if doc_count >= 1 else "#fc8181"
        st.markdown(
            f'<p style="color:{count_color};font-weight:700;font-size:1rem;margin-top:0.5rem;">'
            f"📄 Documents provided: {doc_count} / {len(doc_cols)}"
            f"</p>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown('<p class="section-title">Contact & Region Flags</p>',
                    unsafe_allow_html=True)

        fc1, fc2, fc3 = st.columns(3)
        raw["FLAG_MOBIL"]       = int(fc1.checkbox("Has mobile phone",    value=True))
        raw["FLAG_CONT_MOBILE"] = int(fc2.checkbox("Mobile is reachable", value=True))
        raw["FLAG_PHONE"]       = int(fc3.checkbox("Has home phone"))
        raw["FLAG_EMAIL"]       = int(fc1.checkbox("Has e-mail address"))

        raw["REG_REGION_NOT_LIVE_REGION"] = int(
            fc2.checkbox("Reg. region ≠ live region"))
        raw["REG_REGION_NOT_WORK_REGION"] = int(
            fc3.checkbox("Reg. region ≠ work region"))
        raw["LIVE_REGION_NOT_WORK_REGION"] = int(
            fc1.checkbox("Live region ≠ work region"))

    return raw, prev


def render_results(result: dict) -> None:
    prob       = result["probability"]
    risk_color = result["risk_color"]
    risk_label = result["risk_label"]
    risk_emoji = result["risk_emoji"]
    threshold  = result["threshold"]

    st.markdown("---")
    st.markdown("## 📈 Risk Assessment")

    col_gauge, col_verdict, col_metrics = st.columns([2, 2, 3])

    with col_gauge:
        gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = round(prob * 100, 1),
            number= {"suffix": "%", "font": {"size": 52, "color": risk_color}},
            title = {"text": "Default Probability",
                     "font": {"size": 15, "color": "#94a3b8"}},
            gauge = {
                "axis"     : {"range": [0, 100], "tickwidth": 1,
                               "tickcolor": "#4a5568", "ticksuffix": "%",
                               "tickfont": {"color": "#94a3b8"}},
                "bar"      : {"color": risk_color, "thickness": 0.28},
                "bgcolor"  : "#1a1f2e",
                "borderwidth": 1,
                "bordercolor": "#2d3748",
                "steps": [
                    {"range": [0,  30],  "color": "#1a3a2a"},
                    {"range": [30, 50],  "color": "#2d2a10"},
                    {"range": [50, 70],  "color": "#3d2010"},
                    {"range": [70, 100], "color": "#3d1010"},
                ],
                "threshold": {
                    "line" : {"color": "#e2e8f0", "width": 3},
                    "thickness": 0.82,
                    "value": threshold * 100,
                },
            },
        ))
        gauge.update_layout(
            height=300, margin=dict(t=40, b=0, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font=dict(family="sans-serif", color="#cbd5e0"),
        )
        st.plotly_chart(gauge, use_container_width=True)

    with col_verdict:
        decision_text = "⛔ LOAN AT RISK" if result["prediction"] == 1 else "✅ LOAN APPROVED"
        decision_bg   = "#1f1014" if result["prediction"] == 1 else "#0f1f14"
        decision_border = risk_color

        st.markdown(
            f"""
<div style="background:{decision_bg};border-left:6px solid {decision_border};
            border-radius:10px;padding:1.4rem;margin-top:0.5rem;
            border:1px solid #2d3748;border-left:6px solid {decision_border};">
  <div style="font-size:0.75rem;color:#718096;text-transform:uppercase;
              letter-spacing:0.1em;margin-bottom:0.4rem;">Decision</div>
  <div style="font-size:1.6rem;font-weight:900;color:{decision_border};
              margin-bottom:0.8rem;">{decision_text}</div>
  <div style="font-size:0.75rem;color:#718096;text-transform:uppercase;
              letter-spacing:0.08em;margin-bottom:0.3rem;">Risk Tier</div>
  <div style="font-size:1.1rem;font-weight:700;color:{risk_color};">
    {risk_emoji} {risk_label}
  </div>
  <hr style="border:none;border-top:1px solid #2d3748;margin:0.8rem 0;">
  <div style="display:flex;justify-content:space-between;">
    <span style="color:#718096;font-size:0.82rem;">Score</span>
    <span style="font-weight:700;font-size:0.9rem;color:{risk_color};">{prob:.4f}</span>
  </div>
  <div style="display:flex;justify-content:space-between;">
    <span style="color:#718096;font-size:0.82rem;">Threshold</span>
    <span style="font-weight:700;font-size:0.9rem;color:#e2e8f0;">{threshold:.4f}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_metrics:
        p = result["perf"]
        st.markdown(
            '<p class="section-title" style="margin-top:0.5rem;">Model Metrics (Test Set)</p>',
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        m1.metric("ROC-AUC",       f'{p["test_roc_auc"]:.3f}')
        m2.metric("Avg Precision", f'{p["test_avg_precision"]:.3f}')
        m1.metric("F1 Score",      f'{p["test_f1_score"]:.3f}')
        m2.metric("MCC",           f'{p["test_mcc"]:.3f}')

        d = result["dataset_info"]
        st.caption(
            f"Trained on **{d['train_samples_smote']:,}** SMOTE-balanced samples "
            f"| {d['n_features']} features | default rate {d['fraud_rate_pct']:.1f}%"
        )

    st.markdown("---")
    col_imp, col_model = st.columns([3, 2])

    with col_imp:
        st.markdown("### 🔍 Top-10 Feature Importances")
        st.caption("How much each feature contributed to the model's decision (global, not local).")

        top_feats = result["top_features"]
        feat_names  = list(top_feats.keys())
        feat_values = list(top_feats.values())

        bar_colors = [risk_color if v == max(feat_values) else "#4299e1" for v in feat_values]

        fig_imp = go.Figure(go.Bar(
            x             = feat_values[::-1],
            y             = feat_names[::-1],
            orientation   = "h",
            marker_color  = bar_colors[::-1],
            marker_line_color = "rgba(0,0,0,0.15)",
            marker_line_width = 1,
            text          = [f"{v:.4f}" for v in feat_values[::-1]],
            textposition  = "outside",
            textfont      = {"size": 11},
        ))
        fig_imp.update_layout(
            height=380,
            margin=dict(t=10, b=10, l=10, r=60),
            xaxis=dict(title="Importance Score", title_font=dict(color="#94a3b8"),
                       showgrid=True, gridcolor="#2d3748", zeroline=False,
                       tickfont=dict(color="#94a3b8")),
            yaxis=dict(tickfont=dict(size=12, color="#cbd5e0")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(0,0,0,0)",
            font=dict(color="#cbd5e0"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_model:
        st.markdown("### 🏆 All Model Comparison")
        if result.get("all_models"):
            model_rows = []
            for name, metrics in result["all_models"].items():
                model_rows.append({
                    "Model":        name,
                    "ROC-AUC":     round(metrics.get("ROC-AUC",       0), 4),
                    "F1":           round(metrics.get("F1 Score",       0), 4),
                    "Avg Precision":round(metrics.get("Avg Precision",  0), 4),
                    "MCC":          round(metrics.get("MCC",            0), 4),
                })
            comp_df = pd.DataFrame(model_rows).sort_values("ROC-AUC", ascending=False)
            st.dataframe(
                comp_df.set_index("Model"),
                use_container_width=True,
                height=360,
            )

    st.markdown("---")
    st.markdown("### 🛡️ Rule-Based Risk Audit")

    rule        = result["rule"]
    driver      = result["driver"]
    ml_prob     = result["ml_prob"]
    ml_label    = result["ml_label"]
    ml_color    = result["ml_color"]
    ml_emoji    = result["ml_emoji"]

    driver_cfg = {
        "rule":  ("Rule Engine Override",  "#E53E3E", "⚠️",
                  "The ML model predicted LOW risk, but the rule engine detected hard red flags. "
                  "**Rule-based result takes priority.**"),
        "model": ("ML Model Decision",     "#4299e1", "🤖",
                  "The rule engine found no hard red flags. The ML model's probability score drove the final verdict."),
        "both":  ("Both Systems Agree — HIGH RISK", "#DD6B20", "🔴",
                  "Both the ML model and the rule engine independently flagged this application as high risk."),
        "none":  ("Both Systems Agree — LOW RISK",  "#38A169", "✅",
                  "Both the ML model and the rule engine indicate low default risk for this application."),
    }
    d_title, d_color, d_icon, d_desc = driver_cfg[driver]
    st.markdown(
        f"""
<div style="background:#0a0d14;border-left:5px solid {d_color};border-radius:8px;
            padding:1rem 1.2rem;margin-bottom:1rem;border:1px solid #2d3748;
            border-left:5px solid {d_color};">
  <div style="font-size:1rem;font-weight:700;color:{d_color};">{d_icon} {d_title}</div>
  <div style="font-size:0.85rem;color:#94a3b8;margin-top:0.3rem;">{d_desc}</div>
</div>""",
        unsafe_allow_html=True,
    )

    col_ml, col_rule = st.columns(2)

    with col_ml:
        st.markdown(
            f"""<div style="background:#1a1f2e;border-radius:8px;padding:0.9rem 1rem;
                            border:1px solid #2d3748;text-align:center;">
  <div style="font-size:0.72rem;color:#718096;text-transform:uppercase;
              letter-spacing:0.1em;margin-bottom:0.3rem;">🤖 ML Model (Random Forest)</div>
  <div style="font-size:1.3rem;font-weight:800;color:{ml_color};">{ml_emoji} {ml_label}</div>
  <div style="font-size:0.9rem;color:#94a3b8;margin-top:0.2rem;">
    Probability: <b style="color:{ml_color};">{ml_prob:.4f}</b>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

    with col_rule:
        st.markdown(
            f"""<div style="background:#1a1f2e;border-radius:8px;padding:0.9rem 1rem;
                            border:1px solid #2d3748;text-align:center;">
  <div style="font-size:0.72rem;color:#718096;text-transform:uppercase;
              letter-spacing:0.1em;margin-bottom:0.3rem;">🛡️ Rule Engine</div>
  <div style="font-size:1.3rem;font-weight:800;color:{rule['risk_color']};">{rule['risk_emoji']} {rule['risk_label']}</div>
  <div style="font-size:0.9rem;color:#94a3b8;margin-top:0.2rem;">
    Rule score: <b style="color:{rule['risk_color']};">{rule['score']:.4f}</b>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
    if rule["rules_fired"]:
        sev_icon  = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}
        sev_color = {"HIGH": "#fc8181", "MEDIUM": "#f6e05e", "LOW": "#90cdf4"}
        with st.expander(
            f"📋 {len(rule['rules_fired'])} rule(s) triggered — click to see details",
            expanded=(driver in ("rule", "both")),
        ):
            for r in rule["rules_fired"]:
                icon  = sev_icon.get(r["sev"],  "⚪")
                color = sev_color.get(r["sev"], "#e2e8f0")
                st.markdown(
                    f'<div style="padding:0.45rem 0.7rem;margin-bottom:0.35rem;'
                    f'background:#0f1117;border-radius:6px;border-left:3px solid {color};">'
                    f'<span style="color:{color};font-weight:700;font-size:0.8rem;">'
                    f'{icon} {r["sev"]}</span>'
                    f'<span style="color:#cbd5e0;font-size:0.85rem;margin-left:0.5rem;">'
                    f'{r["msg"]}</span></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.success("✅ No rule violations detected — all financial ratios and scores are within safe bounds.")

    st.markdown("---")
    with st.expander("📖 What do these results mean?", expanded=False):
        st.markdown(
            f"""
**Default Probability: `{prob:.4f}` ({prob*100:.1f}%)**

The model estimates a **{prob*100:.1f}%** chance this applicant will default on their loan.

**Decision threshold: `{threshold:.4f}`**

This threshold was optimised on the validation set to maximise **F1 Score**, balancing
precision and recall for the minority (default) class. If the probability exceeds the
threshold the model flags the application as high-risk.

**Risk Tier Definitions**

| Tier | Probability | Interpretation |
|---|---|---|
| 🟢 LOW RISK         | < 30% | Low likelihood of default — standard approval process |
| 🟡 MODERATE RISK    | 30–50% | Borderline — may require additional documentation |
| 🟠 HIGH RISK        | 50–70% | Elevated default risk — manual underwriter review recommended |
| 🔴 VERY HIGH RISK   | ≥ 70% | Strong indicator of default — consider rejection or higher rate |

**Top Features**

The bar chart shows the **global** feature importances from the Random Forest model. These
represent how much each feature contributes to predictions on average across all training
samples — they are not explanations specific to this individual applicant.
"""
        )


def main() -> None:
    _, _, _, _, _, metadata, _ = load_artifacts()

    render_sidebar(metadata)

    st.markdown(
        """
<div style="background:linear-gradient(135deg,#1a2340,#2b6cb0);
            padding:1.8rem 2rem;border-radius:14px;margin-bottom:1.5rem;
            box-shadow:0 4px 20px rgba(0,0,0,0.18);">
  <h1 style="color:white;margin:0;font-size:2rem;font-weight:800;">
    🏦 Credit Default Risk Analyzer
  </h1>
  <p style="color:#90cdf4;margin:0.4rem 0 0;font-size:1rem;">
    Home Credit Default Detection · Random Forest + SMOTE · Professional ML Pipeline
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    raw_input, prev_app = render_input_form()

    st.markdown("<br>", unsafe_allow_html=True)
    btn_col, _ = st.columns([2, 5])
    analyze = btn_col.button("🔍 Analyze Risk", use_container_width=True)

    if analyze:
        with st.spinner("Running inference…"):
            try:
                result = predict_fraud(raw_input, prev_app)
                render_results(result)
            except Exception as exc:
                st.error(f"❌ Inference error: {exc}")
                with st.expander("Traceback"):
                    import traceback
                    st.code(traceback.format_exc())
    else:
        st.markdown(
            """
<div style="background:#1a1f2e;border-radius:12px;padding:2rem;text-align:center;
            border:2px dashed #2d3748;margin-top:1rem;">
  <p style="font-size:1.3rem;color:#94a3b8;margin:0;">
    ⬆️  Fill in the form above and click <strong style="color:#63b3ed;">Analyze Risk</strong>
  </p>
  <p style="font-size:0.9rem;color:#4a5568;margin:0.4rem 0 0;">
    All fields have sensible defaults — you can click Analyze immediately to see an example result.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
