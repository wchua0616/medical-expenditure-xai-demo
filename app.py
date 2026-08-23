from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from demo_utils import (
    DEFAULT_LABEL_MAPS,
    FEATURE_COLUMNS,
    DISPLAY_NAMES,
    make_input_row,
    explain_prediction,
    make_local_waterfall,
    contribution_table,
)

st.set_page_config(
    page_title="Explainable Healthcare Expenditure Predictor",
    page_icon="📊",
    layout="wide",
)

MODEL_PATH = Path(__file__).with_name("medical_expenditure_demo.pkl")

st.title("Explainable Healthcare Expenditure Predictor")
st.caption(
    "Research demonstration based on the final 14-predictor XGBoost pipeline."
)

st.info(
    "This tool estimates annual individual healthcare expenditure using patterns "
    "learned from the 2023 U.S. MEPS analytical sample. It is not an insurance "
    "premium quotation, medical advice, or a validated tool for non-U.S. populations."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found: `medical_expenditure_demo.pkl`.\n\n"
        "Run the final export cell in the supplied demo-ready notebook, then place "
        "the generated `.pkl` file in the same folder as `app.py`."
    )
    st.stop()

@st.cache_resource
def load_artifact():
    # Only load pickle/joblib files that you created or trust.
    return joblib.load(MODEL_PATH)

artifact = load_artifact()
pipeline = artifact["pipeline"]
label_maps = artifact.get("label_maps", DEFAULT_LABEL_MAPS)
model_name = artifact.get("model_name", "XGBoost")
runtime_versions = artifact.get("versions", {})
global_importance = artifact.get("global_shap_importance")
model_comparison = artifact.get("model_comparison")

st.subheader("Enter the 14 inputs")

left, right = st.columns(2)

with left:
    age = st.number_input(
        "1. Age",
        min_value=18,
        max_value=85,
        value=40,
        step=1,
        help="The MEPS adult analytical sample uses age 18 and above; age is top-coded at 85.",
    )

    sex_label = st.selectbox(
        "2. Sex",
        list(label_maps["sex"].values()),
    )

    race_label = st.selectbox(
        "3. Race / ethnicity",
        list(label_maps["race_ethnicity"].values()),
    )

    poverty_label = st.selectbox(
        "4. Poverty status",
        list(label_maps["poverty_status"].values()),
    )

    insurance_label = st.selectbox(
        "5. Insurance coverage",
        list(label_maps["insurance_coverage"].values()),
    )

    health_label = st.selectbox(
        "6. Self-reported general health",
        list(label_maps["self_reported_health"].values()),
        index=2,
    )

    smoker_label = st.selectbox(
        "7. Currently smoke",
        list(label_maps["currently_smoke"].values()),
        index=1,
    )

with right:
    if smoker_label == "No":
        smoking_frequency_label = "Not at all"
        st.selectbox(
            "8. Smoking frequency",
            ["Not at all"],
            index=0,
            disabled=True,
            help="Set to 'Not at all' when current smoking status is No.",
        )
    else:
        smoking_frequency_label = st.selectbox(
            "8. Smoking frequency",
            ["Every day", "Some days"],
        )

    height_cm = st.number_input(
        "9. Height (cm)",
        min_value=120.0,
        max_value=220.0,
        value=170.0,
        step=0.5,
    )

    weight_kg = st.number_input(
        "10. Weight (kg)",
        min_value=30.0,
        max_value=250.0,
        value=70.0,
        step=0.5,
    )

    # BMI is calculated behind the scenes for the trained model.
    bmi = float(weight_kg) / ((float(height_cm) / 100.0) ** 2)
    bmi_missing = 0

    hbp_label = st.selectbox(
        "11. Diagnosed with high blood pressure",
        list(label_maps["high_blood_pressure_dx"].values()),
        index=1,
    )

    diabetes_label = st.selectbox(
        "12. Diagnosed with diabetes",
        list(label_maps["diabetes_dx"].values()),
        index=1,
    )

    cancer_label = st.selectbox(
        "13. Diagnosed with cancer",
        list(label_maps["cancer_dx"].values()),
        index=1,
    )

    adl_label = st.selectbox(
        "14. Need help with Activities of Daily Living",
        list(label_maps["adl_help_needed"].values()),
        index=1,
        help=(
            "Activities of Daily Living include basic self-care activities "
            "such as bathing, dressing, eating, toileting and moving around."
        ),
    )


def reverse_lookup(mapping, label):
    for code, name in mapping.items():
        if name == label:
            return code
    raise ValueError(f"Unknown label: {label}")


if st.button("Predict healthcare expenditure", type="primary", use_container_width=True):
    raw_values = {
        "age": float(age),
        "bmi": float(bmi),
        "sex": reverse_lookup(label_maps["sex"], sex_label),
        "race_ethnicity": reverse_lookup(label_maps["race_ethnicity"], race_label),
        "poverty_status": reverse_lookup(label_maps["poverty_status"], poverty_label),
        "insurance_coverage": reverse_lookup(label_maps["insurance_coverage"], insurance_label),
        "self_reported_health": reverse_lookup(label_maps["self_reported_health"], health_label),
        "currently_smoke": reverse_lookup(label_maps["currently_smoke"], smoker_label),
        "smoking_frequency": reverse_lookup(
            label_maps["smoking_frequency"], smoking_frequency_label
        ),
        "high_blood_pressure_dx": reverse_lookup(
            label_maps["high_blood_pressure_dx"], hbp_label
        ),
        "diabetes_dx": reverse_lookup(label_maps["diabetes_dx"], diabetes_label),
        "cancer_dx": reverse_lookup(label_maps["cancer_dx"], cancer_label),
        "adl_help_needed": reverse_lookup(label_maps["adl_help_needed"], adl_label),
        "bmi_missing": int(bmi_missing),
    }

    X_user = make_input_row(raw_values)

    prediction = float(pipeline.predict(X_user)[0])

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric(
        "Estimated annual healthcare expenditure",
        f"${prediction:,.0f}",
    )
    metric_col2.metric("Model", model_name)
    metric_col3.metric("Predictors used", "14")

    st.caption(
        f"BMI calculated from entered height and weight: {bmi:.2f} kg/m²"
    )

    if prediction < 0:
        st.warning(
            "The unconstrained XGBoost model produced a negative estimate. "
            "A practical zero-floor sensitivity would report $0, but the SHAP "
            "breakdown below explains the original model output."
        )

    with st.expander("Input profile"):
        profile = pd.DataFrame(
            {
                "Predictor": [DISPLAY_NAMES[c] for c in FEATURE_COLUMNS],
                "Entered value": [
                    (
{
                            "age": age,
                            "bmi": round(bmi, 2),
                            "sex": sex_label,
                            "race_ethnicity": race_label,
                            "poverty_status": poverty_label,
                            "insurance_coverage": insurance_label,
                            "self_reported_health": health_label,
                            "currently_smoke": smoker_label,
                            "smoking_frequency": smoking_frequency_label,
                            "high_blood_pressure_dx": hbp_label,
                            "diabetes_dx": diabetes_label,
                            "cancer_dx": cancer_label,
                            "adl_help_needed": adl_label,
                            "bmi_missing": "No",
                        }.get(c)
                    )
                    for c in FEATURE_COLUMNS
                ],
            }
        )
        st.dataframe(profile, use_container_width=True, hide_index=True)

    with st.spinner("Calculating local SHAP explanation..."):
        explanation = explain_prediction(
            pipeline=pipeline,
            X_raw=X_user,
            raw_values=raw_values,
            label_maps=label_maps,
        )

    st.subheader("Why did the model produce this estimate?")
    st.caption(
        "Positive SHAP contributions increase this individual's prediction relative "
        "to the model baseline; negative contributions reduce it. Contributions are "
        "aggregated back to the 14 original predictors."
    )

    fig = make_local_waterfall(
        base_value=explanation["base_value"],
        prediction=explanation["prediction"],
        contributions=explanation["contributions"],
        predictor_labels=explanation["predictor_labels"],
    )
    st.pyplot(fig, use_container_width=True)

    table = contribution_table(
        explanation["contributions"],
        explanation["predictor_labels"],
    )
    st.dataframe(
        table.style.format({"SHAP contribution (USD)": "${:,.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Technical checks"):
        st.write(
            f"SHAP reconstruction error: "
            f"{explanation['reconstruction_error']:.6f}"
        )
        st.write(
            "Baseline + summed SHAP contributions = "
            f"${explanation['base_value'] + sum(explanation['contributions'].values()):,.2f}"
        )
        st.write(f"Pipeline prediction = ${explanation['prediction']:,.2f}")
        if runtime_versions:
            st.json(runtime_versions)

    if global_importance:
        with st.expander("Global SHAP importance from the held-out test sample"):
            global_df = pd.DataFrame(global_importance)
            st.dataframe(global_df, use_container_width=True, hide_index=True)

    if model_comparison:
        with st.expander("Final model comparison stored with the artifact"):
            st.dataframe(
                pd.DataFrame(model_comparison),
                use_container_width=True,
                hide_index=True,
            )
