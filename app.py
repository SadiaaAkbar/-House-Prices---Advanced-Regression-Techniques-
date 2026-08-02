import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="California House Price Predictor", page_icon="🏠", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("model/house_price_best_pipeline.joblib")


model = load_model()

st.title("🏠 California House Price Predictor")
st.write(
    "Predicts median house value for a California block group using a "
    "**Random Forest** model (R² ≈ 0.73, RMSE ≈ $61,300 on held-out test data), "
    "the best performer out of Linear Regression, Random Forest, and XGBoost — "
    "trained on the classic California Housing dataset."
)

st.divider()
st.subheader("Enter block-group details")

col1, col2 = st.columns(2)

with col1:
    median_income = st.number_input(
        "Median income (10,000s of $, e.g. 5.5 = $55,000)",
        min_value=0.0, max_value=20.0, value=5.0, step=0.1,
        help="Median household income in the block group, in tens of thousands of dollars.",
    )
    housing_median_age = st.number_input(
        "Median house age (years)",
        min_value=0, max_value=60, value=25, step=1,
    )
    population = st.number_input(
        "Population",
        min_value=1, max_value=40000, value=1200, step=10,
        help="Total number of people living in the block group.",
    )

with col2:
    total_rooms = st.number_input(
        "Total rooms",
        min_value=1, max_value=40000, value=2500, step=10,
        help="Total rooms across all households in the block group.",
    )
    total_bedrooms = st.number_input(
        "Total bedrooms",
        min_value=1, max_value=8000, value=500, step=10,
    )
    households = st.number_input(
        "Households",
        min_value=1, max_value=6000, value=450, step=10,
        help="Number of households (used to derive rooms per household).",
    )

ocean_proximity = st.selectbox(
    "Ocean proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
)

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    if total_rooms == 0 or households == 0:
        st.error("Total rooms and households must be greater than 0.")
    else:
        rooms_per_household = total_rooms / households
        bedrooms_per_room = total_bedrooms / total_rooms

        input_df = pd.DataFrame([{
            "median_income": median_income,
            "total_rooms": total_rooms,
            "housing_median_age": housing_median_age,
            "population": population,
            "total_bedrooms": total_bedrooms,
            "rooms_per_household": rooms_per_household,
            "bedrooms_per_room": bedrooms_per_room,
            "ocean_proximity": ocean_proximity,
        }])

        prediction = model.predict(input_df)[0]

        st.success(f"### Predicted median house value: **${prediction:,.0f}**")

        with st.expander("Derived features used by the model"):
            st.write(f"- Rooms per household: `{rooms_per_household:.2f}`")
            st.write(f"- Bedrooms per room: `{bedrooms_per_room:.2f}`")

st.divider()
st.caption(
    "Model: Random Forest Regressor · Data: California Housing (block-group census data) · "
    "Built for the Neurofive Solutions ML Track"
)
