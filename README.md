# Explainable Healthcare Expenditure Predictor — Demo

This Streamlit demo uses the **final 14-predictor XGBoost pipeline** from the supplied research notebook and produces:

- an annual healthcare-expenditure prediction;
- the user's complete 14-predictor input profile;
- a local SHAP waterfall aggregated back to the **original 14 predictors**;
- a ranked SHAP-contribution table;
- optional stored model-comparison and global-SHAP results.

## Important scope

The research model predicts **individual total healthcare expenditure** using the 2023 U.S. MEPS analytical sample. It is not a medical-inflation forecast, insurance premium quote, medical advice, or a model validated for Malaysia or other non-U.S. populations.

## Model inputs

The fitted pipeline uses the same 14 predictors as the research notebook:

1. age
2. BMI
3. sex
4. race / ethnicity
5. poverty status
6. insurance coverage
7. self-reported health
8. current smoking status
9. smoking frequency
10. high blood pressure diagnosis
11. diabetes diagnosis
12. cancer diagnosis
13. ADL help needed
14. BMI missingness indicator

The BMI-missingness indicator is derived automatically from the "BMI unavailable" checkbox.
