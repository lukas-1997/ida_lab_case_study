import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from icuzen import ICUZen

class ICUApi:
    def __init__(self, base_url: str = "https://idalab-icu.ew.r.appspot.com"):
        self.base_url = base_url

    def get_sample_vital_signs(self):
        try:
            response = requests.get(f"{self.base_url}/sample_vital_signs")
            response.raise_for_status()
            return response.json().get("patient_list", [])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching sample vital signs: {e}")

    def get_history_vital_signs(self):
        try:
            response = requests.get(f"{self.base_url}/history_vital_signs")
            response.raise_for_status()
            return response.json().get("patient_list", [])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching history vital signs: {e}")

    def get_patient_ids(self):
        try:
            response = requests.get(f"{self.base_url}/patient_ids")
            response.raise_for_status()
            return response.json().get("patient_id_list", [])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching patient IDs: {e}")

    def get_patient_vital_signs(self, patient_id: str):
        try:
            response = requests.get(f"{self.base_url}/patient_vital_signs/{patient_id}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error fetching patient vital signs: {e}")

def normalize_vitals(X):
    normal_values = np.array([37.0, 120.0, 80.0, 75.0, 15.0])
    return X / normal_values

@st.cache_resource
def get_model(threshold=0.76, sensitivity=30):
    return ICUZen(threshold=threshold, sensitivity=sensitivity)

def parse_vital_signs(vital_signs_str):
    vital_signs = {}
    for item in vital_signs_str.split(";"):
        key, value = item.split("->")
        vital_signs[key.strip()] = float(value.strip())
    return [
        vital_signs['body_temperature'],
        vital_signs['blood_pressure_systolic'],
        vital_signs['blood_pressure_diastolic'],
        vital_signs['heart_rate'],
        vital_signs['respiratory_rate']
    ]

# Initialize API
api = ICUApi()

# Initialize session state
if "fetched_data" not in st.session_state:
    st.session_state["fetched_data"] = pd.DataFrame()
if "model_params" not in st.session_state:
    st.session_state["model_params"] = {"threshold": 0.76, "sensitivity": 30}

st.title("ICU Alarm Management Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "Fetch Data", "Parameter Heatmap", "Model Analysis", "Historical Data", "System Status"]
)

st.sidebar.header("Model Parameters")
sensitivity = st.sidebar.number_input("Sensitivity", 1, 100, st.session_state["model_params"]["sensitivity"], 1)
threshold = st.sidebar.number_input("Threshold", 0.0, 1.0, st.session_state["model_params"]["threshold"], 0.01)

if (threshold != st.session_state["model_params"]["threshold"] or
    sensitivity != st.session_state["model_params"]["sensitivity"]):
    st.session_state["model_params"]["threshold"] = threshold
    st.session_state["model_params"]["sensitivity"] = sensitivity
    st.cache_resource.clear()

model = get_model(threshold=st.session_state["model_params"]["threshold"],
                 sensitivity=st.session_state["model_params"]["sensitivity"])

if menu == "Overview":
    st.header("ICU Alarm Management System")
    st.write("""
    This system monitors ICU patient vital signs and uses the ICUzen model
    to identify potential false alarms.
    """)

elif menu == "Fetch Data":
    st.header("Data Collection")

    data_source = st.radio("Select Data Source", ["Sample Data", "Historical Data", "Specific Patient"])

    if data_source == "Specific Patient":
        try:
            patient_ids = api.get_patient_ids()
            selected_patient = st.selectbox("Select Patient ID", patient_ids)
            if st.button("Fetch Patient Data"):
                patient_data = api.get_patient_vital_signs(selected_patient)
                if patient_data:
                    vital_signs = parse_vital_signs(patient_data["vital_signs"])
                    df = pd.DataFrame([{
                        'temp': vital_signs[0],
                        'bp_sys': vital_signs[1],
                        'bp_dia': vital_signs[2],
                        'hr': vital_signs[3],
                        'rr': vital_signs[4],
                        'patient_id': patient_data['patient_id'],
                        'timestamp': datetime.now()
                    }])
                    st.session_state["fetched_data"] = df
                    st.success("Data fetched successfully!")
                    st.dataframe(df)
        except Exception as e:
            st.error(f"Error: {e}")

    else:
        col1, col2 = st.columns(2)
        with col1:
            fetch_duration = st.slider("Collection Duration (seconds)", 5, 60, 30)
        with col2:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                required_columns = ['temp', 'bp_sys', 'bp_dia', 'hr', 'rr', 'patient_id']
                if not all(col in df.columns for col in required_columns):
                    st.error("CSV must contain columns: temp, bp_sys, bp_dia, hr, rr, patient_id")
                else:
                    st.session_state["fetched_data"] = df
                    st.success("CSV file uploaded successfully!")
                    st.dataframe(df)
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

        elif st.button("Start Collection"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            start_time = datetime.now()
            collected_data = []

            while (datetime.now() - start_time).seconds < fetch_duration:
                try:
                    if data_source == "Sample Data":
                        new_data = api.get_sample_vital_signs()
                    else:  # Historical Data
                        new_data = api.get_history_vital_signs()

                    if new_data:
                        for patient in new_data:
                            vital_signs = parse_vital_signs(patient["vital_signs"])
                            collected_data.append({
                                'temp': vital_signs[0],
                                'bp_sys': vital_signs[1],
                                'bp_dia': vital_signs[2],
                                'hr': vital_signs[3],
                                'rr': vital_signs[4],
                                'patient_id': patient['patient_id'],
                                'timestamp': datetime.now()
                            })

                    progress = (datetime.now() - start_time).seconds / fetch_duration
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Collecting data... {len(collected_data)} records processed")

                except Exception as e:
                    st.error(f"Error fetching data: {e}")
                    break

            st.session_state["fetched_data"] = pd.DataFrame(collected_data)
            st.success("Data collection complete!")
            st.dataframe(st.session_state["fetched_data"])

elif menu == "Model Analysis":
    st.header("Model Analysis")

    if st.session_state["fetched_data"].empty:
        st.warning("No data available. Please fetch data first.")
    else:
        X = np.array(st.session_state["fetched_data"][['temp', 'bp_sys', 'bp_dia', 'hr', 'rr']])
        X_normalized = normalize_vitals(X)
        predictions = model.predict(X_normalized)

        st.session_state["fetched_data"]['is_critical'] = predictions

        st.subheader("Analysis Results")
        st.dataframe(st.session_state["fetched_data"])

        fig = px.pie(
            st.session_state["fetched_data"],
            names="is_critical",
            title="Alarm Classification Distribution",
            labels={'is_critical': 'Critical Situation'}
        )
        st.plotly_chart(fig)

        st.subheader("Summary")
        st.write(f"Total records: {len(predictions)}")
        st.write(f"Critical situations: {sum(predictions)}")
        st.write(f"Non-critical situations: {len(predictions) - sum(predictions)}")

elif menu == "Parameter Heatmap":
    st.header("Parameter Sensitivity Analysis")

    if st.session_state["fetched_data"].empty:
        st.warning("No data available. Please fetch data first.")
    else:
        sensitivity_range = np.linspace(1, 100, 20)
        threshold_range = np.linspace(0, 1, 20)

        alarm_rates = []
        X = np.array(st.session_state["fetched_data"][['temp', 'bp_sys', 'bp_dia', 'hr', 'rr']])
        X_normalized = normalize_vitals(X)

        for sens in sensitivity_range:
            row = []
            for thresh in threshold_range:
                test_model = ICUZen(threshold=thresh, sensitivity=sens)
                predictions = test_model.predict(X_normalized)
                alarm_rate = (np.sum(predictions) / len(predictions)) * 100
                row.append(alarm_rate)
            alarm_rates.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=alarm_rates,
            x=threshold_range,
            y=sensitivity_range,
            hoverongaps=False,
            hovertemplate="Sensitivity: %{y:.1f}<br>Threshold: %{x:.2f}<br>Critical Rate: %{z:.1f}%<extra></extra>",
            colorscale="RdBu_r"
        ))

        fig.update_layout(
            title="Alarm Classification Rate Heatmap",
            xaxis_title="Threshold",
            yaxis_title="Sensitivity",
            width=800,
            height=600
        )

        st.plotly_chart(fig)

        st.write(f"Current parameters: Threshold={model.threshold:.2f}, Sensitivity={model.sensitivity:.1f}")
        current_rate = (np.sum(model.predict(X_normalized)) / len(X)) * 100
        st.write(f"Current critical situation rate: {current_rate:.1f}%")

elif menu == "Historical Data":
    st.header("Historical Data")
    if not st.session_state["fetched_data"].empty:
        csv = st.session_state["fetched_data"].to_csv(index=False)
        st.download_button(
            "Download Data",
            csv,
            "icu_data.csv",
            "text/csv"
        )
    else:
        st.info("No data available.")

elif menu == "System Status":
    st.header("System Status")
    st.write("Current Model Configuration:")
    st.write(f"- Threshold: {model.threshold}")
    st.write(f"- Sensitivity: {model.sensitivity}")

    if not st.session_state["fetched_data"].empty:
        st.write(f"Collected records: {len(st.session_state['fetched_data'])}")