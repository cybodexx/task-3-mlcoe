
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


REFERENCE_CATEGORY = "Other / training baseline"
CATEGORY_OPTION_LIMIT = 5
CATEGORY_GROUPS = {
    "Personal and education": [
        "education",
        "sex",
    ],
    "Work": [
        "class_of_worker",
        "major_industry_code",
        "major_occupation_code",
    ],
    "Tax filing": [
        "tax_filer_status",
    ],
}
NUMERIC_FEATURES = {
    "occupation_code": ("Occupation code", 0),
    "capital_gain": ("Capital gains ($)", 0),
    "capital_loss": ("Capital losses ($)", 0),
    "dividends_from_stocks": ("Dividends from stocks ($)", 0),
    "num_persons_worked_for": ("People worked for", 0),
    "wage_per_hour": ("Wage per hour ($)", 0),
}


def sanitize_feature_name(name):
    return re.sub(r"[<>\[\]]", "", str(name))


@st.cache_resource
def load_model_files():
    model_dir = Path(__file__).resolve().parent
    model = joblib.load(model_dir / "best_model.pkl")
    saved_columns = joblib.load(model_dir / "model_columns.pkl")
    columns = [sanitize_feature_name(column) for column in saved_columns]
    expected_columns = list(getattr(model, "feature_names_in_", columns))
    if columns != expected_columns:
        raise ValueError("Saved model columns do not match the trained model feature names.")
    return model, columns


model, feature_columns = load_model_files()
feature_importance = dict(zip(feature_columns, model.feature_importances_))
st.title("Census Income Prediction")
st.write("Estimate whether a person's income is $50,000 or more using the strongest model features.")

with st.form("income_prediction"):
    st.subheader("Numeric details")
    numeric_values = {}
    numeric_columns = st.columns(2)
    numeric_values["age"] = numeric_columns[0].slider("Age", 0, 100, 35)
    numeric_values["weeks_worked"] = numeric_columns[1].slider("Weeks worked", 0, 52, 40)

    for index, (feature, (label, default)) in enumerate(NUMERIC_FEATURES.items()):
        if feature in feature_columns:
            numeric_values[feature] = numeric_columns[index % 2].number_input(
                label,
                min_value=0,
                value=default,
                step=1,
                key=f"numeric_{feature}",
            )

    selected_categories = {}
    for group_label, prefixes in CATEGORY_GROUPS.items():
        with st.expander(group_label, expanded=(group_label == "Personal and education")):
            category_columns = st.columns(2)
            for index, prefix in enumerate(prefixes):
                encoded_prefix = f"{prefix}_"
                options = sorted(
                    [column for column in feature_columns if column.startswith(encoded_prefix)],
                    key=lambda column: feature_importance.get(column, 0),
                    reverse=True,
                )[:CATEGORY_OPTION_LIMIT]
                options = [
                    column[len(encoded_prefix):]
                    for column in feature_columns
                    if column in options
                ]
                if options:
                    selected_categories[prefix] = category_columns[index % 2].selectbox(
                        prefix.replace("_", " ").title(),
                        [REFERENCE_CATEGORY, *options],
                        key=f"category_{prefix}",
                    )

    predict = st.form_submit_button("Predict income")

if predict:
    row = {column: 0 for column in feature_columns}
    row.update(numeric_values)
    row["has_capital_gains"] = int(numeric_values["capital_gain"] > 0)

    age_group = pd.cut(
        [numeric_values["age"]],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
    )[0]
    if not pd.isna(age_group):
        age_group_column = sanitize_feature_name(f"age_group_{age_group}")
        if age_group_column in row:
            row[age_group_column] = 1

    for prefix, category in selected_categories.items():
        if category != REFERENCE_CATEGORY:
            encoded_column = sanitize_feature_name(f"{prefix}_{category}")
            if encoded_column in row:
                row[encoded_column] = 1

    input_data = pd.DataFrame([row], columns=feature_columns)
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    if prediction == 1:
        st.success("Predicted income: $50,000 or more")
    else:
        st.info("Predicted income: Less than $50,000")
    st.metric("Estimated probability of $50,000+ income", f"{probability:.1%}")

