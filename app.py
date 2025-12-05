import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration ---
st.set_page_config(
    page_title="Drug Effectiveness Predictor",
    page_icon="💊",
    layout="centered"
)

# --- Load Custom CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Try to load the CSS file
if os.path.exists('style.css'):
    local_css("style.css")

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    try:
        # Load the specific filenames provided
        model = joblib.load('drug_model.pkl')
        # Note: The file name provided was "scalar.pkl"
        scaler = joblib.load('scalar.pkl')
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}. Please ensure 'drug_model.pkl' and 'scalar.pkl' are in the same directory.")
        return None, None

model, scaler = load_assets()

# --- Main Interface ---
st.title("💊 Drug Effectiveness Predictor")
st.markdown("Enter the patient details and drug information below to predict the effectiveness score.")

if model and scaler:
    # --- Define Feature Columns ---
    # These match the columns found in your scalar.pkl EXACTLY
    expected_columns = [
        'Dosage_mg', 
        'Drug_Name_Sertraline', 'Drug_Name_Ciprofloxacin', 
        'Condition_Diabetes', 
        'Age', 
        'Drug_Name_Metformin', 'Drug_Name_Bupropion', 'Drug_Name_Amoxicillin', 'Drug_Name_Losartan', 
        'Condition_Pain Relief', 
        'Drug_Name_Glipizide', 'Drug_Name_Ibuprofen', 'Drug_Name_Tramadol', 'Drug_Name_Escitalopram', 
        'Treatment_Duration_days', 
        'Condition_Hypertension', 
        'Drug_Name_Metoprolol', 
        'Condition_Infection', 
        'Drug_Name_Insulin Glargine', 'Drug_Name_Paracetamol', 'Drug_Name_Azithromycin'
    ]

    # --- Extract Options for Dropdowns ---
    # We strip the prefixes to make clean dropdown lists
    drug_options = [col.replace('Drug_Name_', '') for col in expected_columns if 'Drug_Name_' in col]
    condition_options = [col.replace('Condition_', '') for col in expected_columns if 'Condition_' in col]

    # --- Input Form ---
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Patient Details")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            condition = st.selectbox("Medical Condition", options=condition_options)
            duration = st.number_input("Treatment Duration (Days)", min_value=1, value=10)

        with col2:
            st.subheader("Medication Details")
            drug_name = st.selectbox("Drug Name", options=drug_options)
            dosage = st.number_input("Dosage (mg)", min_value=1.0, value=500.0, step=10.0)

        submitted = st.form_submit_button("Predict Effectiveness")

    # --- Prediction Logic ---
    if submitted:
        # 1. Initialize a dataframe with zeros for all expected columns
        input_data = pd.DataFrame(np.zeros((1, len(expected_columns))), columns=expected_columns)

        # 2. Fill in numerical values
        input_data['Age'] = age
        input_data['Dosage_mg'] = dosage
        input_data['Treatment_Duration_days'] = duration

        # 3. Fill in One-Hot Encoded values
        # We construct the column name (e.g., 'Drug_Name_Ibuprofen') and set it to 1
        drug_col = f"Drug_Name_{drug_name}"
        condition_col = f"Condition_{condition}"

        if drug_col in expected_columns:
            input_data[drug_col] = 1
        
        if condition_col in expected_columns:
            input_data[condition_col] = 1

        try:
            # 4. Scale the data
            input_scaled = scaler.transform(input_data)

            # 5. Predict
            prediction = model.predict(input_scaled)[0]

            # 6. Display Result
            st.markdown("---")
            st.success("Prediction Complete")
            
            # Creating a visual score card
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.metric(label="Effectiveness Score", value=f"{prediction:.2f}")
            
            with col_res2:
                # Interpret the score (Example logic - adjust based on your specific metric range)
                if prediction > 7:
                    st.markdown('<div class="result-card high-score">High Effectiveness</div>', unsafe_allow_html=True)
                    st.write("The model predicts this treatment is highly effective for the given parameters.")
                elif prediction > 4:
                    st.markdown('<div class="result-card med-score">Moderate Effectiveness</div>', unsafe_allow_html=True)
                    st.write("The model predicts moderate effectiveness.")
                else:
                    st.markdown('<div class="result-card low-score">Low Effectiveness</div>', unsafe_allow_html=True)
                    st.write("The model predicts low effectiveness. Consider reviewing dosage or drug choice.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
            st.write("Debug info - Input shape:", input_data.shape)

else:
    st.warning("Please upload 'drug_model.pkl' and 'scalar.pkl' to the directory.")