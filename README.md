# Diabète Predict — Clinical ML Diagnostic & Power BI Suite

End-to-end **Healthcare Machine Learning & Business Intelligence** system combining SMOTE-balanced gradient boosting (`CatBoost` / `LightGBM` / `Random Forest`, **87.1% ROC-AUC**) deployed in a bilingual **EN / FR Clinical Streamlit Application** alongside a companion **Power BI (`.pbix`) cohort analytics dashboard**.

---

## Key Capabilities

- **Class-Imbalance Clinical Pipeline**: Median physiological zero imputation, `RobustScaler`, `SMOTE` oversampling, and `GridSearchCV` hyperparameter optimization trained on the 768-patient Pima clinical dataset.
- **Bilingual EN / FR Clinical Streamlit Interface (`app/`)**:
  - Real-time diabetes risk probability gauge & biomarker reference comparison chart.
  - Quick clinical reference presets (Low-Risk vs High-Risk profiles) plus live cohort explorer (`768` patient records).
- **Power BI Executive Companion (`bi/Diabete_Predict.pbix`)**: Cohort analytics exploring Glucose, BMI, Age, and Diabetes Pedigree correlations.

---

## Project Structure

```text
Diabete-Predict-Clinical-ML-PowerBI/
├── app/
│   ├── app.py               # Bilingual (EN/FR) Streamlit diagnostic & cohort UI
│   ├── sidebar.py           # Validated model metrics & clinical disclaimer sidebar
│   └── style.css            # Clinical Blue & Mint Teal theme styles
├── bi/
│   └── Diabete_Predict.pbix # Power BI Desktop executive workbook
├── data/
│   └── diabetes.csv         # Clinical cohort dataset (768 patients)
├── docs/
│   ├── presentation.pdf     # Exported project slide deck (PDF)
│   ├── presentation.pptx    # Original slide deck
│   ├── Rapport.pdf          # Full technical report
│   └── screenshots/
├── models/
│   ├── model_smote.pkl      # Trained serialized classifier
│   └── scaler.pkl           # Fitted feature scaler
├── notebooks/
│   └── Diabete_predict.ipynb
├── .streamlit/config.toml
├── .gitignore
└── requirements.txt
```

---

## Quick Start

```powershell
cd D:\work\Projects\Diabete-Predict-Clinical-ML-PowerBI
pip install -r requirements.txt
streamlit run app/app.py
```
Opens locally at `http://localhost:8501`.
