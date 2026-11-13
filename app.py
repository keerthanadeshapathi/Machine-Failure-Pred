import streamlit as st
import requests
st.set_page_config(page_title="Machine Failure Prediction", page_icon="⚙️")
API_URL = "https://machine-failure-pred-3.onrender.com/predict"
st.title("⚙️ Machine Failure Prediction")
st.write("Enter machine details to predict whether the machine will fail.")
machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
air_temperature = st.number_input("Air Temperature (K)", value=300.0)
process_temperature = st.number_input("Process Temperature (K)", value=310.0)
rotational_speed = st.number_input("Rotational Speed (rpm)", value=1500)
torque = st.number_input("Torque (Nm)", value=40.0)
tool_wear = st.number_input("Tool Wear (min)", value=100)
if st.button("Predict Machine Failure"):
    input_data = {
        "Type": machine_type,
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear
    }

    try:
        response = requests.post(API_URL, json=input_data, timeout=120)

        if response.status_code == 200:
            result = response.json()

            if result["prediction"] == 1:
                st.error("⚠️ Machine Failure")
            else:
                st.success("✅ No Machine Failure")

            st.write("Prediction:", result["prediction"])
        else:
            st.error(f"API Error: {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
