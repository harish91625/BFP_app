import streamlit as st
import joblib
import numpy as np

bfp_model= joblib.load("bfp_eff_design.sav")
st.title("Design BFP efficiency")
st.write("Enter the BFP flow to know the design BFP Efficiency")

flow_value= st.number_input("Enter Flow:",min_value=0.0, max_value=1000.0, value=100.0, step=1.0)

if st.button("Design BFP Efficiency"):
    input_array= np.array([[flow_value]])
    output= bfp_model.predict(input_array)
    st.success(f"Design BFP Efficiency for the flow of {flow_value} is : **{output[0]:.2f}%**")


st.info("This model uses Random Forest to predict BFP efficiency based on flow rate")
