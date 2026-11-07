import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load trained model and feature columns
model = joblib.load("model.joblib")
feature_columns = joblib.load("feature_columns.joblib")
metrics = pd.read_csv("model_metrics.csv")
data = pd.read_csv("dataset/Housing.csv")

# Title
st.title("🏠 House Price Prediction")
st.write("Machine learning project using regression algorithms")

# Navigation
page = st.sidebar.radio(
    "Choose Page",
    ["Dataset Overview", "Predict House Price", "Algorithm Comparison"]
)

# Dataset Overview
if page == "Dataset Overview":

    st.header("Dataset Overview")

    st.write("Housing dataset details")

    st.dataframe(data.head(), width="stretch")

    st.write("Dataset Shape:", data.shape)

    st.write("Dataset Information")

    st.write(data.describe())

# Prediction Page
elif page == "Predict House Price":

    st.header("Enter House Details")

    st.write("Choose values for the features below.")

    area = st.number_input(
        "Area (square feet)", min_value=100, value=4600
    )

    bedrooms = st.number_input(
        "Bedrooms", min_value=1, value=3
    )

    bathrooms = st.number_input(
        "Bathrooms", min_value=1, value=2
    )

    stories = st.number_input(
        "Stories", min_value=1, value=2
    )

    parking = st.number_input(
        "Parking", min_value=0, value=1
    )

    mainroad = st.selectbox("Mainroad", ["yes", "no"])
    guestroom = st.selectbox("Guestroom", ["yes", "no"])
    basement = st.selectbox("Basement", ["yes", "no"])
    hotwaterheating = st.selectbox(
        "Hot Water Heating", ["yes", "no"]
    )
    airconditioning = st.selectbox(
        "Air Conditioning", ["yes", "no"]
    )
    prefarea = st.selectbox("Preferred Area", ["yes", "no"])

    furnishingstatus = st.selectbox(
        "Furnishing Status",
        ["furnished", "semi-furnished", "unfurnished"]
    )

    if st.button("Predict Price", type="primary"):

        input_data = pd.DataFrame(
            0, index=[0], columns=feature_columns
        )

        input_data["area"] = area
        input_data["bedrooms"] = bedrooms
        input_data["bathrooms"] = bathrooms
        input_data["stories"] = stories
        input_data["parking"] = parking

        categorical_values = {
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus
        }

        for column, value in categorical_values.items():
            feature_name = f"{column}_{value}"

            if feature_name in input_data.columns:
                input_data[feature_name] = 1

        prediction = model.predict(input_data)[0]

        st.success("Prediction Successful!")

        st.subheader("Estimated House Price")

        st.title(f"₹ {prediction:,.2f}")

        st.info(
            "This is a model estimate based on the training "
            "dataset, not a guaranteed market price."
        )

        with st.expander("View the input values"):
            st.dataframe(input_data, width="stretch")

# Algorithm Comparison
elif page == "Algorithm Comparison":

    st.header("Algorithm Comparison")

    st.write("Performance of different regression algorithms")

    st.dataframe(metrics, width="stretch")

    st.subheader("RMSE Comparison")

    st.bar_chart(
        metrics.set_index("Model")["RMSE"]
    )

    st.subheader("R2 Score Comparison")

    st.bar_chart(
        metrics.set_index("Model")["R2"]
    )

    best_model = metrics.loc[metrics["RMSE"].idxmin()]

    st.success(
        f"Best Model: {best_model['Model']}"
    )