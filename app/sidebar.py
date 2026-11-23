import streamlit as st

SIDEBAR_COPY = {
    "en": {
        "lang_label": "Interface Language",
        "brand_title": "Diabete Predict",
        "brand_sub": "Clinical Risk Scoring & BI Support",
        "perf_heading": "Validated Model Metrics",
        "auc_text": "Confidence Score (ROC-AUC): 87.1%",
        "acc": "Accuracy",
        "auc": "ROC-AUC",
        "rec": "Recall",
        "prec": "Precision",
        "stack_heading": "Pipeline Architecture",
        "stack_items": "- CatBoost & LightGBM Ensemble\n- SMOTE Class Balancing\n- RobustScaler + GridSearchCV\n- Power BI Cohort BI",
        "notice_heading": "Clinical Disclaimer",
        "notice_body": "This ML tool supports clinical exploration and prevention awareness; it does not replace formal medical diagnosis.",
        "compliance": "Built for reproducible ML evaluation",
    },
    "fr": {
        "lang_label": "Langue de l'interface",
        "brand_title": "Diabète Predict",
        "brand_sub": "Scoring Clinique & Aide Décisionnelle",
        "perf_heading": "Métriques Validées du Modèle",
        "auc_text": "Score de confiance (ROC-AUC) : 87.1%",
        "acc": "Exactitude",
        "auc": "ROC-AUC",
        "rec": "Rappel",
        "prec": "Précision",
        "stack_heading": "Architecture du Pipeline",
        "stack_items": "- Ensemble CatBoost & LightGBM\n- Rééquilibrage SMOTE\n- RobustScaler + GridSearchCV\n- Tableau de bord Power BI",
        "notice_heading": "Mention Clinique",
        "notice_body": "Cet outil ML accompagne l'analyse préventive et ne remplace pas une consultation médicale personnalisée.",
        "compliance": "Pipeline reproductible & traçable",
    },
}


def show_sidebar() -> str:
    with st.sidebar:
        lang_choice = st.radio(
            "Language / Langue",
            options=["FR", "EN"],
            horizontal=True,
            index=0,
        )
        lang = lang_choice.lower()
        txt = SIDEBAR_COPY[lang]

        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg,#0c4a6e,#0d9488);color:#fff;padding:1rem;border-radius:12px;margin:0.5rem 0 1rem 0;">
              <div style="font-size:1.15rem;font-weight:800;">{txt['brand_title']}</div>
              <div style="font-size:0.8rem;opacity:0.9;">{txt['brand_sub']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"#### {txt['perf_heading']}")
        st.progress(0.871, text=txt["auc_text"])

        c1, c2 = st.columns(2)
        c1.metric(txt["acc"], "85.7%")
        c2.metric(txt["auc"], "87.1%")
        c1.metric(txt["rec"], "83.2%")
        c2.metric(txt["prec"], "79.5%")

        st.divider()
        st.markdown(f"#### {txt['stack_heading']}")
        st.info(txt["stack_items"])

        st.markdown(f"#### {txt['notice_heading']}")
        st.warning(txt["notice_body"])
        st.caption(txt["compliance"])

        return lang
