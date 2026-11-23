from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from sidebar import show_sidebar

ROOT_DIR = APP_DIR.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "data" / "diabetes.csv"

st.set_page_config(
    page_title="Diabete Predict — Clinical ML Diagnostic",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css(css_path: Path):
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

load_css(APP_DIR / "style.css")
lang = show_sidebar()

COPY = {
    "fr": {
        "badge": "Système d'Aide au Diagnostic ML // SMOTE + Gradient Boosting",
        "kpi_cohort": "Taille Cohorte",
        "kpi_auc": "ROC-AUC",
        "kpi_acc": "Exactitude",
        "kpi_rec": "Rappel",
        "title": "Diabète Predict — Plateforme d'Analyse Clinique",
        "sub": "Évaluation prédictive du risque diabétique à partir de constantes biométriques (cohorte Pima 768 patientes), couplée au suivi décisionnel Power BI.",
        "tabs": ["Évaluation Patient", "Explorateur de Cohorte (768 Patient(e)s)", "Protocole & Architecture ML"],
        "form_title": "Saisie des Constantes Cliniques du Patient",
        "form_hint": "Ajustez les paramètres biologiques et biométriques ci-dessous puis lancez l'inférence.",
        "col1_title": "Constantes Biologiques",
        "col2_title": "Paramètres Biométriques",
        "preg": "Nombre de grossesses",
        "glu": "Glucose plasmatique (mg/dL)",
        "bp": "Pression diastolique (mm Hg)",
        "skin": "Épaisseur du pli cutané (mm)",
        "ins": "Insuline sérique (mu U/ml)",
        "bmi": "Indice de Masse Corporelle (IMC)",
        "dpf": "Score de prédisposition familiale",
        "age": "Âge (années)",
        "btn_predict": "Générer l'Analyse Diagnostique",
        "preset_lbl": "Profils rapides de test",
        "preset_opts": ["Personnalisé", "Profil Risque Faible", "Profil Risque Élevé"],
        "res_title": "Résultat de l'Évaluation Clinique",
        "high_title": "Risque Élevé Détecté",
        "low_title": "Risque Faible / Rassurant",
        "gauge_title": "Probabilité Estimée (%)",
        "factor_title": "Positionnement Biométrique vs Seuils Cliniques",
        "rec_high_head": "Recommandations Cliniques Préventives",
        "rec_low_head": "Maintien des Bonnes Constantes",
        "rec_high": [
            "Planifier un bilan glycémique complet (HbA1c et glycémie à jeun).",
            "Consulter un professionnel de santé pour validation clinique.",
            "Surveiller régulièrement la pression artérielle et l'IMC.",
            "Privilégier une alimentation à index glycémique contrôlé et une activité régulière.",
        ],
        "rec_low": [
            "Maintenir un suivi préventif périodique selon l'âge.",
            "Conserver des habitudes nutritionnelles équilibrées et une activité physique régulière.",
            "Surveiller les constantes métaboliques annuellement.",
        ],
        "cohort_title": "Analyse Exploratoire de la Cohorte Pima (768 Dossiers)",
        "cohort_scatter": "Distribution Glucose vs IMC par Statut Diagnostique",
        "cohort_hist": "Distribution de l'Âge par Résultat",
        "outcome_0": "Négatif (0)",
        "outcome_1": "Positif (1)",
        "arch_models": "Modèles Comparés",
        "arch_models_txt": "CatBoost, LightGBM, Random Forest, Régression Logistique évalués par validation croisée.",
        "arch_prep": "Prétraitement Clinique",
        "arch_prep_txt": "Imputation médiane des zéros physiologiques, RobustScaler, rééquilibrage SMOTE et GridSearchCV.",
        "arch_perf": "Performances Validées",
        "arch_perf_txt": "Accuracy : 85.7% | ROC-AUC : 87.1% | Rappel : 83.2% | Précision : 79.5%",
        "footer": "Outil de démonstration Machine Learning — Consultez toujours un professionnel de santé pour tout avis médical.",
    },
    "en": {
        "badge": "Clinical ML Decision Support // SMOTE + Gradient Boosting",
        "kpi_cohort": "Cohort Size",
        "kpi_auc": "ROC-AUC",
        "kpi_acc": "Accuracy",
        "kpi_rec": "Recall",
        "title": "Diabete Predict — Clinical Risk Assessment Platform",
        "sub": "Predictive diabetes risk scoring trained on biometric patient cohorts (768 Pima records), paired with an analytical Power BI executive dashboard.",
        "tabs": ["Patient Risk Assessment", "Cohort Explorer (768 Patients)", "ML Protocol & Architecture"],
        "form_title": "Patient Clinical Parameters",
        "form_hint": "Adjust biological and biometric indicators below, or pick a clinical preset profile, then run inference.",
        "col1_title": "Biological Measurements",
        "col2_title": "Biometric Indicators",
        "preg": "Pregnancies Count",
        "glu": "Plasma Glucose (mg/dL)",
        "bp": "Diastolic Blood Pressure (mm Hg)",
        "skin": "Triceps Skinfold Thickness (mm)",
        "ins": "Serum Insulin (mu U/ml)",
        "bmi": "Body Mass Index (BMI)",
        "dpf": "Diabetes Pedigree Function",
        "age": "Patient Age (years)",
        "btn_predict": "Run Diagnostic Risk Analysis",
        "preset_lbl": "Quick Test Presets",
        "preset_opts": ["Custom", "Low-Risk Reference Profile", "High-Risk Clinical Profile"],
        "res_title": "Diagnostic Risk Assessment Output",
        "high_title": "Elevated Risk Flagged",
        "low_title": "Low Clinical Risk",
        "gauge_title": "Estimated Risk Probability (%)",
        "factor_title": "Patient Biomarkers vs Clinical Reference Thresholds",
        "rec_high_head": "Recommended Clinical Follow-Up",
        "rec_low_head": "Preventive Care Guidance",
        "rec_high": [
            "Schedule a confirmatory HbA1c and fasting plasma glucose panel.",
            "Review findings directly with a licensed clinician.",
            "Track blood pressure and BMI progression longitudinally.",
            "Adopt controlled-glycemic nutrition and structured physical activity.",
        ],
        "rec_low": [
            "Maintain routine annual preventive metabolic screenings.",
            "Sustain balanced nutrition and regular aerobic/strength exercise.",
            "Track biometric trends over time.",
        ],
        "cohort_title": "Pima Cohort Exploratory Analytics (768 Records)",
        "cohort_scatter": "Glucose vs BMI Distribution by Diagnostic Outcome",
        "cohort_hist": "Age Distribution by Clinical Outcome",
        "outcome_0": "Negative (0)",
        "outcome_1": "Positive (1)",
        "arch_models": "Benchmarked Classifiers",
        "arch_models_txt": "CatBoost, LightGBM, Random Forest, and Logistic Regression evaluated via stratified cross-validation.",
        "arch_prep": "Clinical Preprocessing",
        "arch_prep_txt": "Physiological zero imputation, RobustScaler normalization, SMOTE oversampling, and GridSearchCV.",
        "arch_perf": "Validated Benchmarks",
        "arch_perf_txt": "Accuracy: 85.7% | ROC-AUC: 87.1% | Recall: 83.2% | Precision: 79.5%",
        "footer": "Machine Learning demonstration tool — Always consult a qualified medical professional for diagnosis.",
    },
}

t = COPY[lang]


@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "model_smote.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    return model, scaler


@st.cache_data
def load_cohort():
    df = pd.read_csv(DATA_PATH)
    df["Diagnostic"] = df["Outcome"].map({0: t["outcome_0"], 1: t["outcome_1"]})
    return df


# Inline clinical SVG icon (no external CDN dependency)
CLINICAL_SVG = """
<svg width="56" height="56" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="56" height="56" rx="14" fill="rgba(255,255,255,0.16)" stroke="rgba(255,255,255,0.35)"/>
  <path d="M16 28H22L25.5 19L30.5 37L34 28H40" stroke="#FFFFFF" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

st.markdown(
    f"""
    <div class="clinical-header">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:1.5rem;flex-wrap:wrap;">
        <div style="flex:1;min-width:260px;">
          <div class="clinical-badge">{t['badge']}</div>
          <h1 style="margin:0;font-size:2rem;font-weight:800;">{t['title']}</h1>
          <p style="margin:0.45rem 0 0 0;opacity:0.92;font-size:0.95rem;">{t['sub']}</p>
        </div>
        <div>{CLINICAL_SVG}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

cohort_df = load_cohort()
m1, m2, m3, m4 = st.columns(4)
m1.markdown(f'<div class="metric-tile"><div class="metric-title">{t["kpi_cohort"]}</div><div class="metric-value">{len(cohort_df)}</div></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="metric-tile"><div class="metric-title">{t["kpi_auc"]}</div><div class="metric-value">87.1%</div></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="metric-tile"><div class="metric-title">{t["kpi_acc"]}</div><div class="metric-value">85.7%</div></div>', unsafe_allow_html=True)
m4.markdown(f'<div class="metric-tile"><div class="metric-title">{t["kpi_rec"]}</div><div class="metric-value">83.2%</div></div>', unsafe_allow_html=True)

st.write("")
tab_score, tab_cohort, tab_arch = st.tabs(t["tabs"])

with tab_score:
    st.subheader(t["form_title"])
    st.caption(t["form_hint"])

    preset = st.selectbox(t["preset_lbl"], t["preset_opts"], index=0)
    defaults = {
        t["preset_opts"][0]: (1, 108, 72, 22, 85, 26.2, 0.42, 31),
        t["preset_opts"][1]: (1, 92, 68, 18, 70, 23.4, 0.25, 26),
        t["preset_opts"][2]: (4, 168, 84, 34, 195, 36.8, 0.92, 49),
    }[preset]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{t['col1_title']}**")
        pregnancies = st.slider(t["preg"], 0, 17, defaults[0])
        glucose = st.slider(t["glu"], 40, 200, defaults[1])
        blood_pressure = st.slider(t["bp"], 24, 122, defaults[2])
        skin_thickness = st.slider(t["skin"], 7, 99, defaults[3])

    with col2:
        st.markdown(f"**{t['col2_title']}**")
        insulin = st.slider(t["ins"], 14, 846, defaults[4])
        bmi = st.slider(t["bmi"], 15.0, 67.1, float(defaults[5]))
        diabetes_pedigree = st.slider(t["dpf"], 0.05, 2.42, float(defaults[6]))
        age = st.slider(t["age"], 21, 81, defaults[7])

    run_pred = st.button(t["btn_predict"], type="primary", width="stretch")

    if run_pred or preset != t["preset_opts"][0]:
        model, scaler = load_model()
        feature_cols = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        input_df = pd.DataFrame(
            [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]],
            columns=feature_cols,
        )
        input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_cols)
        prediction = int(model.predict(input_scaled)[0])
        probability = float(model.predict_proba(input_scaled)[0, 1]) * 100.0

        st.divider()
        st.subheader(t["res_title"])
        r_left, r_right = st.columns([1, 1.3])

        with r_left:
            is_high = prediction == 1
            bg_grad = "linear-gradient(135deg,#fee2e2,#fecaca)" if is_high else "linear-gradient(135deg,#dcfce7,#bbf7d0)"
            border_c = "#dc2626" if is_high else "#0d9488"
            text_c = "#991b1b" if is_high else "#065f46"
            head_txt = t["high_title"] if is_high else t["low_title"]

            st.markdown(
                f"""
                <div style="background:{bg_grad};border:2px solid {border_c};border-radius:14px;padding:1.2rem;text-align:center;margin-bottom:0.8rem;">
                  <div style="color:{text_c};font-size:1.1rem;font-weight:800;">{head_txt}</div>
                  <div style="color:{text_c};font-size:2.6rem;font-weight:900;margin:0.3rem 0;">{probability:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability,
                    number={"suffix": "%"},
                    title={"text": t["gauge_title"], "font": {"size": 14}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#dc2626" if is_high else "#0369a1"},
                        "steps": [
                            {"range": [0, 40], "color": "#ccfbf1"},
                            {"range": [40, 70], "color": "#fef3c7"},
                            {"range": [70, 100], "color": "#fee2e2"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="#ffffff")
            st.plotly_chart(fig_gauge, width="stretch")

        with r_right:
            factors = pd.DataFrame({
                "Biomarker": ["Glucose (vs 140)", "BMI (vs 30)", "Blood Pressure (vs 80)", "Age (vs 45)"],
                "Normalized Ratio (%)": [
                    round((glucose / 140.0) * 100, 1),
                    round((bmi / 30.0) * 100, 1),
                    round((blood_pressure / 80.0) * 100, 1),
                    round((age / 45.0) * 100, 1),
                ],
            })
            fig_bar = px.bar(
                factors,
                x="Normalized Ratio (%)",
                y="Biomarker",
                orientation="h",
                template="plotly_white",
                color="Normalized Ratio (%)",
                color_continuous_scale=["#0d9488", "#0369a1", "#dc2626"],
                title=t["factor_title"],
            )
            fig_bar.add_vline(x=100, line_dash="dash", line_color="#64748b")
            fig_bar.update_layout(height=240, margin=dict(l=10, r=10, t=35, b=10), coloraxis_showscale=False, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
            st.plotly_chart(fig_bar, width="stretch")

            rec_head = t["rec_high_head"] if is_high else t["rec_low_head"]
            rec_list = t["rec_high"] if is_high else t["rec_low"]
            bullets = "".join(f"<li>{item}</li>" for item in rec_list)
            st.markdown(
                f"""<div class="clinical-card"><strong>{rec_head}</strong><ul style="margin:0.5rem 0 0 0;padding-left:1.2rem;">{bullets}</ul></div>""",
                unsafe_allow_html=True,
            )

with tab_cohort:
    st.subheader(t["cohort_title"])
    ch1, ch2 = st.columns(2)
    with ch1:
        clean_plot_df = cohort_df[(cohort_df["Glucose"] > 0) & (cohort_df["BMI"] > 0)]
        fig_sc = px.scatter(
            clean_plot_df,
            x="Glucose",
            y="BMI",
            color="Diagnostic",
            template="plotly_white",
            color_discrete_map={t["outcome_0"]: "#0d9488", t["outcome_1"]: "#dc2626"},
            opacity=0.72,
            title=t["cohort_scatter"],
        )
        fig_sc.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        st.plotly_chart(fig_sc, width="stretch")

    with ch2:
        fig_age = px.histogram(
            cohort_df,
            x="Age",
            color="Diagnostic",
            barmode="overlay",
            template="plotly_white",
            color_discrete_map={t["outcome_0"]: "#0369a1", t["outcome_1"]: "#ef4444"},
            title=t["cohort_hist"],
        )
        fig_age.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
        st.plotly_chart(fig_age, width="stretch")

with tab_arch:
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(f'<div class="clinical-card"><h4>{t["arch_models"]}</h4><p>{t["arch_models_txt"]}</p></div>', unsafe_allow_html=True)
    with a2:
        st.markdown(f'<div class="clinical-card"><h4>{t["arch_prep"]}</h4><p>{t["arch_prep_txt"]}</p></div>', unsafe_allow_html=True)
    with a3:
        st.markdown(f'<div class="clinical-card"><h4>{t["arch_perf"]}</h4><p>{t["arch_perf_txt"]}</p></div>', unsafe_allow_html=True)

st.divider()
st.caption(t["footer"])
