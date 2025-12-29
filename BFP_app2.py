import streamlit as st
import joblib
import numpy as np

# Load models
bfp_model = joblib.load("bfp_eff_design.sav")
bfp_model1 = joblib.load("bfp_head_design.sav")

st.title("Design BFP EFFICIENCY and HEAD")
st.write("Enter the BFP flow to know the design BFP Efficiency and BFP HEAD")

# Input
flow_value = st.number_input(
    "Enter Flow:",
    min_value=0.0,
    max_value=1000.0,
    value=100.0,
    step=1.0
)

# Button and prediction
if st.button("Design BFP Efficiency and Head"):
    input_array = np.array([[flow_value]])
    output = bfp_model.predict(input_array)
    output1 = bfp_model1.predict(input_array)

    st.success(
        f"Design BFP Efficiency and Head for the test flow of {flow_value} "
        f"is : **{output[0]:.2f}%** and {output1[0]:.2f}"
    )

    # Image at the bottom of the output
    st.image(
        "BFP Design eff.png",              # <--- put your image file name here
        caption="BFP Design Curve",
        use_column_width=True
    )

st.info(
    "This model uses Random Forest machine learning model "
    "to get BFP design efficiency and design head based on flow rate"
)
