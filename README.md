# Explainable Healthcare Expenditure Predictor — Demo

This Streamlit demo uses the **final 14-predictor XGBoost pipeline** from the supplied research notebook and produces:

- an annual healthcare-expenditure prediction;
- the user's input profile, with BMI calculated automatically from height and weight;
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
13. Activities of Daily Living help needed
14. BMI missingness indicator

Users do not enter BMI directly. The app asks for height (cm) and weight (kg), calculates BMI behind the scenes, and sets the BMI-missingness indicator to 0.


## User-facing inputs

The app asks for 14 user-facing inputs:

1. Age
2. Sex
3. Race / ethnicity
4. Poverty status
5. Insurance coverage
6. Self-reported general health
7. Current smoking status
8. Smoking frequency
9. Height (cm)
10. Weight (kg)
11. High blood pressure diagnosis
12. Diabetes diagnosis
13. Cancer diagnosis
14. Need help with Activities of Daily Living

BMI is calculated behind the scenes as:

`BMI = weight (kg) / [height (m)]²`

The trained model still receives the same original BMI predictor. Because height and weight are required in the demo, the BMI-missingness indicator is fixed at 0.
