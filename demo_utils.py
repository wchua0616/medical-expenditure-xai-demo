import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FEATURE_COLUMNS = [
    "age",
    "bmi",
    "sex",
    "race_ethnicity",
    "poverty_status",
    "insurance_coverage",
    "self_reported_health",
    "currently_smoke",
    "smoking_frequency",
    "high_blood_pressure_dx",
    "diabetes_dx",
    "cancer_dx",
    "adl_help_needed",
    "bmi_missing",
]

DISPLAY_NAMES = {
    "age": "Age",
    "bmi": "BMI",
    "sex": "Sex",
    "race_ethnicity": "Race / Ethnicity",
    "poverty_status": "Poverty Status",
    "insurance_coverage": "Insurance Coverage",
    "self_reported_health": "Self-Reported Health",
    "currently_smoke": "Current Smoking Status",
    "smoking_frequency": "Smoking Frequency",
    "high_blood_pressure_dx": "High Blood Pressure",
    "diabetes_dx": "Diabetes",
    "cancer_dx": "Cancer",
    "adl_help_needed": "Activities of Daily Living Help Needed",
    "bmi_missing": "BMI Missing Indicator",
}

DEFAULT_LABEL_MAPS = {
    "sex": {1: "Male", 2: "Female"},
    "race_ethnicity": {
        1: "White",
        2: "Black",
        3: "Amer. Indian / Alaska Native",
        4: "Asian / Others",
        6: "Multiple Races",
    },
    "poverty_status": {
        1: "Poor",
        2: "Near Poor",
        3: "Low Income",
        4: "Middle Income",
        5: "High Income",
    },
    "insurance_coverage": {
        1: "Any Private",
        2: "Public Only",
        3: "Uninsured",
    },
    "self_reported_health": {
        1: "Excellent",
        2: "Very Good",
        3: "Good",
        4: "Fair",
        5: "Poor",
    },
    "currently_smoke": {1: "Yes", 2: "No"},
    "smoking_frequency": {
        1: "Every day",
        2: "Some days",
        3: "Not at all",
    },
    "high_blood_pressure_dx": {1: "Yes", 2: "No"},
    "diabetes_dx": {1: "Yes", 2: "No"},
    "cancer_dx": {1: "Yes", 2: "No"},
    "adl_help_needed": {1: "Yes", 2: "No"},
}


def make_input_row(raw_values):
    missing = [c for c in FEATURE_COLUMNS if c not in raw_values]
    if missing:
        raise ValueError(f"Missing predictors: {missing}")
    return pd.DataFrame(
        [{c: raw_values[c] for c in FEATURE_COLUMNS}],
        columns=FEATURE_COLUMNS,
    )


def _clean_feature_names(preprocessor):
    names = preprocessor.get_feature_names_out()
    clean = []
    for name in names:
        if "__" in name:
            clean.append(name.split("__", 1)[1])
        else:
            clean.append(name)
    return clean


def _map_encoded_to_original(encoded_name):
    if encoded_name in FEATURE_COLUMNS:
        return encoded_name

    # Longest names first prevents accidental prefix collisions.
    for original in sorted(FEATURE_COLUMNS, key=len, reverse=True):
        if encoded_name.startswith(original + "_"):
            return original

    raise ValueError(f"Could not map encoded feature: {encoded_name}")


def _value_label(feature, raw_values, label_maps):
    value = raw_values[feature]

    if feature == "age":
        return f"Age: {float(value):.0f}"

    if feature == "bmi":
        if pd.isna(value):
            return "BMI: Missing"
        return f"BMI: {float(value):.1f}"

    if feature == "bmi_missing":
        return f"BMI missing indicator: {'Yes' if int(value) == 1 else 'No'}"

    mapping = label_maps.get(feature)
    if mapping is not None:
        label = mapping.get(int(value), str(value))
        return f"{DISPLAY_NAMES[feature]}: {label}"

    return f"{DISPLAY_NAMES[feature]}: {value}"


def explain_prediction(pipeline, X_raw, raw_values, label_maps=None):
    label_maps = label_maps or DEFAULT_LABEL_MAPS

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X_raw)
    clean_names = _clean_feature_names(preprocessor)

    # Preferred route: SHAP TreeExplainer.
    try:
        import shap

        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
        )
        shap_exp = explainer(X_transformed)

        values = np.asarray(shap_exp.values)
        if values.ndim == 1:
            shap_values = values
        else:
            shap_values = values[0]

        base_values = np.asarray(shap_exp.base_values).reshape(-1)
        base_value = float(base_values[0])

    except Exception:
        # Robust fallback: XGBoost's native SHAP contributions.
        import xgboost as xgb

        dmatrix = xgb.DMatrix(X_transformed)
        contrib = model.get_booster().predict(
            dmatrix,
            pred_contribs=True,
        )[0]

        shap_values = np.asarray(contrib[:-1], dtype=float)
        base_value = float(contrib[-1])

    if len(clean_names) != len(shap_values):
        raise ValueError(
            f"Feature-name count ({len(clean_names)}) does not match "
            f"SHAP-value count ({len(shap_values)})."
        )

    contributions = {feature: 0.0 for feature in FEATURE_COLUMNS}

    for encoded_name, shap_value in zip(clean_names, shap_values):
        original = _map_encoded_to_original(encoded_name)
        contributions[original] += float(shap_value)

    prediction = float(pipeline.predict(X_raw)[0])
    reconstructed = base_value + sum(contributions.values())

    predictor_labels = {
        feature: _value_label(feature, raw_values, label_maps)
        for feature in FEATURE_COLUMNS
    }

    return {
        "prediction": prediction,
        "base_value": base_value,
        "contributions": contributions,
        "predictor_labels": predictor_labels,
        "reconstruction_error": abs(prediction - reconstructed),
    }


def contribution_table(contributions, predictor_labels):
    rows = []
    for feature, value in contributions.items():
        rows.append(
            {
                "Predictor": predictor_labels[feature],
                "SHAP contribution (MYR)": float(value),
                "Direction": "Increases prediction" if value >= 0 else "Decreases prediction",
            }
        )

    return (
        pd.DataFrame(rows)
        .assign(_abs=lambda x: x["SHAP contribution (MYR)"].abs())
        .sort_values("_abs", ascending=False)
        .drop(columns="_abs")
        .reset_index(drop=True)
    )


def make_local_waterfall(
    base_value,
    prediction,
    contributions,
    predictor_labels,
):
    # Sort by absolute contribution so the strongest drivers are easiest to read.
    items = sorted(
        contributions.items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )

    names = [predictor_labels[k] for k, _ in items]
    values = np.array([v for _, v in items], dtype=float)

    starts = []
    ends = []
    current = float(base_value)

    for value in values:
        starts.append(current)
        current += value
        ends.append(current)

    starts = np.asarray(starts)
    ends = np.asarray(ends)

    fig, ax = plt.subplots(figsize=(11, 8))
    y = np.arange(len(values))

    low = min(base_value, prediction, starts.min(), ends.min())
    high = max(base_value, prediction, starts.max(), ends.max())
    span = max(high - low, 1.0)
    padding = span * 0.012

    positive_color = "#ff2b5b"
    negative_color = "#4285f4"

    for i, (value, start, end) in enumerate(zip(values, starts, ends)):
        left = min(start, end)
        width = abs(value)
        color = positive_color if value >= 0 else negative_color

        ax.barh(
            i,
            width,
            left=left,
            height=0.66,
            color=color,
        )

        if value >= 0 and width > span * 0.09:
            ax.text(
                left + width / 2,
                i,
                f"{value:+,.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
            )
        else:
            # Negative labels are intentionally placed to the right of blue bars.
            label_x = max(start, end) + padding
            ax.text(
                label_x,
                i,
                f"{value:+,.2f}",
                ha="left",
                va="center",
                color=color,
                fontsize=9,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()

    ax.axvline(base_value, linestyle="--", linewidth=1, color="grey")
    ax.axvline(prediction, linestyle="--", linewidth=1, color="grey")

    ax.text(
        base_value,
        -0.055,
        f"E[f(X)] = ${base_value:,.2f}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
        color="grey",
        clip_on=False,
    )

    ax.text(
        prediction,
        1.015,
        f"f(x) = ${prediction:,.2f}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=9,
        color="grey",
        clip_on=False,
    )

    ax.set_xlabel("Healthcare Expenditure Prediction (MYR/year)")
    ax.set_title("Local SHAP Explanation – XGBoost")
    ax.grid(axis="y", linestyle=":", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.92)
    return fig
