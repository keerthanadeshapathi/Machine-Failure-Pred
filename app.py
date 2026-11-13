import streamlit as st
import requests
# Page configuration
st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️"
)
# Title
st.title("⚙️ Machine Failure Prediction")
st.write(
    "Enter machine details to predict whether the machine will fail."
)
# Machine inputs
machine_type = st.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)
air_temperature = st.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=300.0
)
process_temperature = st.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=310.0
)
rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=0.0,
    value=1500.0
)
torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    value=40.0
)
tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0.0,
    value=100.0
)
# Prediction button
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
        response = requests.post(
           "https://machine-failure-pred-7wuu.onrender.com/predict",
            json=input_data,
            timeout=15
        )
        if response.status_code == 200:
            result = response.json()
            if result["prediction"] == 1:
                st.error("⚠️ Machine Failure Predicted!")
            else:
                st.success("✅ No Machine Failure Predicted!")
            st.write(
                "Prediction:",
                result["prediction"]
            )
        else:
            st.error("Prediction Failed!")
            st.write("Status Code:", response.status_code)
            st.write("Error:", response.text)
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. Please start the backend."
        )
    except requests.exceptions.Timeout:
        st.error("API request timed out.")
    except Exception as e:
        st.error(f"Error: {e}")
