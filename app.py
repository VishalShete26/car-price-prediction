import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-family: 'Poppins', sans-serif;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 22px;
        background: linear-gradient(120deg, #10243a, #1e4970);
        color: #ffffff;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: #ffffff;
        font-size: clamp(1.8rem, 4vw, 2.6rem);
        font-weight: 800;
        margin: 0 0 .45rem 0;
    }

    .hero p {
        color: #d8e6f3;
        margin: 0;
        font-size: .96rem;
    }

    .section-heading {
        font-size: 1.12rem;
        font-weight: 700;
        margin: .8rem 0 .7rem 0;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 18px;
        border: 1px solid rgba(90, 160, 120, .45);
        background: rgba(50, 150, 95, .09);
        margin-top: 1.2rem;
    }

    .result-label {
        font-size: .9rem;
        opacity: .8;
        margin-bottom: .35rem;
    }

    .result-price {
        font-size: clamp(2rem, 5vw, 3rem);
        font-weight: 800;
        line-height: 1.2;
        color: #169b62;
    }

    .result-subtitle {
        font-size: .9rem;
        opacity: .8;
        margin-top: .45rem;
    }

    div.stButton > button {
        width: 100%;
        min-height: 3.1rem;
        border-radius: 11px;
        font-weight: 700;
    }

    [data-testid="stForm"] {
        border: 1px solid rgba(128, 128, 128, .25);
        border-radius: 16px;
        padding: 1.2rem;
    }

    footer {visibility: hidden;}

    @media (max-width: 640px) {
        .block-container {
            padding: 1rem 1rem 2rem 1rem;
        }
        .hero {
            padding: 1.35rem;
            border-radius: 16px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# LOAD MODEL BUNDLE
# ---------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "car_price_model.pkl"


@st.cache_resource
def load_model_bundle(model_path: str):
    with open(model_path, "rb") as file:
        bundle = pickle.load(file)

    required_keys = {
        "model",
        "label_encoders",
        "categorical_columns",
        "feature_columns",
    }

    if not isinstance(bundle, dict):
        raise TypeError(
            "The pickle file must contain a model bundle dictionary. "
            "Please recreate it using the model-saving code."
        )

    missing_keys = required_keys.difference(bundle.keys())
    if missing_keys:
        raise KeyError(
            f"Missing keys in model bundle: {', '.join(sorted(missing_keys))}"
        )

    return bundle


try:
    bundle = load_model_bundle(str(MODEL_PATH))
    model = bundle["model"]
    label_encoders = bundle["label_encoders"]
    categorical_columns = bundle["categorical_columns"]
    feature_columns = bundle["feature_columns"]
except FileNotFoundError:
    st.error(
        "Model file not found. Place car_price_model.pkl in the same "
        "folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🚗 Car Price Predictor</h1>
        <p>Estimate your car's resale value using a Machine Learning model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Enter the vehicle details below to estimate its selling price. "
    "All price inputs and predictions are in Indian Rupees."
)


# ---------------------------------------------------------
# INPUT FORM
# ---------------------------------------------------------
st.markdown('<div class="section-heading">Vehicle information</div>',
            unsafe_allow_html=True)

with st.form("car_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        car_encoder = label_encoders["Car_Name"]
        car_options = [str(value) for value in car_encoder.classes_]

        car_name = st.selectbox(
            "Car name",
            options=car_options,
            help="Select a car name available in the training dataset.",
        )

        year = st.number_input(
            "Manufacturing year",
            min_value=1990,
            max_value=2035,
            value=2018,
            step=1,
        )

        present_price = st.number_input(
            "Present showroom price (₹ lakh)",
            min_value=0.01,
            max_value=1000.0,
            value=5.0,
            step=0.1,
            format="%.2f",
        )

        kms_driven = st.number_input(
            "Kilometers driven",
            min_value=0,
            max_value=2_000_000,
            value=30_000,
            step=1_000,
        )

    with col2:
        fuel_encoder = label_encoders["Fuel_Type"]
        fuel_type = st.selectbox(
            "Fuel type",
            options=[str(value) for value in fuel_encoder.classes_],
        )

        seller_encoder = label_encoders["Seller_Type"]
        seller_type = st.selectbox(
            "Seller type",
            options=[str(value) for value in seller_encoder.classes_],
        )

        transmission_encoder = label_encoders["Transmission"]
        transmission = st.selectbox(
            "Transmission",
            options=[str(value) for value in transmission_encoder.classes_],
        )

        owner = st.selectbox(
            "Number of previous owners",
            options=[0, 1, 2, 3],
            format_func=lambda value: (
                "First owner" if value == 0
                else f"{value} previous owner(s)"
            ),
        )

    submitted = st.form_submit_button(
        "Predict Selling Price",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if submitted:
    try:
        # Build a single-row input with original column names.
        input_data = {
            "Car_Name": car_name,
            "Year": int(year),
            "Present_Price": float(present_price),
            "Kms_Driven": int(kms_driven),
            "Fuel_Type": fuel_type,
            "Seller_Type": seller_type,
            "Transmission": transmission,
            "Owner": int(owner),
        }

        input_df = pd.DataFrame([input_data])

        # Apply the exact fitted encoders used during training.
        for column in categorical_columns:
            encoder = label_encoders[column]
            selected_value = str(input_df.loc[0, column])

            if selected_value not in encoder.classes_:
                raise ValueError(
                    f"'{selected_value}' is not supported for {column}."
                )

            input_df[column] = encoder.transform(
                input_df[column].astype(str)
            )

        # Keep the exact feature order expected by the model.
        input_df = input_df[feature_columns]

        predicted_lakh = float(model.predict(input_df)[0])

        if predicted_lakh < 0:
            st.warning(
                "The model returned a negative estimate. "
                "Please verify the input details and training data."
            )
        else:
            predicted_rupees = predicted_lakh * 100_000

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Estimated selling price</div>
                    <div class="result-price">₹{predicted_lakh:,.2f} lakh</div>
                    <div class="result-subtitle">
                        Equivalent to ₹{predicted_rupees:,.0f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                "This is an estimated value generated by your "
                "Decision Tree Regression model, not a guaranteed market price."
            )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
st.caption(
    "Car Price Predictor • Decision Tree Regression • "
    "Predictions are estimates and may differ from actual market prices."
)
