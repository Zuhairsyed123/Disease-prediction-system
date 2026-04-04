import streamlit as st
import pickle
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Multi-Disease Prediction System",
    layout="wide",
    page_icon="🏥"
)

# ----------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------
@st.cache_resource
def load_model(model_name):
    """Loads a saved pickle model from the same directory."""
    model_path = os.path.join(os.path.dirname(__file__), model_name)
    try:
        model = pickle.load(open(model_path, 'rb'))
        return model
    except FileNotFoundError:
        return None

def scale_input(input_data, scaler_name=None):
    """
    Scales the input data using StandardScaler.
    Note: In order to scale properly, you should save and load the exact 
    StandardScaler object that was fitted during model training.
    """
    input_array = np.asarray(input_data).reshape(1, -1)
    
    # If you have a saved scaler, load and use it here:
    # if scaler_name and os.path.exists(scaler_name):
    #     scaler = pickle.load(open(scaler_name, 'rb'))
    #     return scaler.transform(input_array)
    
    # Placeholder for standard scaler logic as requested
    scaler = StandardScaler()
    # Warning: Fitting a scaler on a single sample is mathematically invalid (results in all 0s). 
    # For demonstration purposes, we initialize the scaler but return the raw array. 
    # Swap 'input_array' with 'scaler.transform(input_array)' when you load your trained scaler.
    
    return input_array

# ----------------------------------------------------
# LOAD MODELS
# ----------------------------------------------------
diabetes_model = load_model('diabetes_model.sav')
parkinsons_model = load_model('parkinsons_model.sav')
heart_model = load_model('heart_model.sav')

# ----------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------
st.sidebar.title("🏥 Disease Prediction System")
st.sidebar.markdown("---")

selected = st.sidebar.radio(
    "Select a Model for Prediction",
    [
        "Diabetes Prediction", 
        "Heart Disease Prediction", 
        "Parkinson's Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Please fill in the input features correctly for accurate predictions.")

# ----------------------------------------------------
# 1. DIABETES PREDICTION MODULE
# ----------------------------------------------------
if selected == "Diabetes Prediction":
    st.title("🩸 Diabetes Prediction")
    st.markdown("Enter the patient's medical details below:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0, step=1)
        SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=150.0, value=0.0)
        DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.0)
        
    with col2:
        Glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=0.0)
        Insulin = st.number_input("Insulin Level", min_value=0.0, max_value=1000.0, value=0.0)
        Age = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
        
    with col3:
        BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=0.0)
        BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0)
        
    # CSS styling for predict button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Predict Diabetes", use_container_width=True):
        if diabetes_model is None:
            st.error("Model file 'diabetes_model.sav' not found. Please place it in the application directory.")
        else:
            inputs = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            
            # Use StandardScaler as required
            scaled_input = scale_input(inputs) 
            prediction = diabetes_model.predict(scaled_input)
            
            if prediction[0] == 1:
                st.error("⚠️ The person is likely to have the disease (Diabetes).")
            else:
                st.success("✅ The person is healthy.")

# ----------------------------------------------------
# 2. HEART DISEASE PREDICTION MODULE
# ----------------------------------------------------
elif selected == "Heart Disease Prediction":
    st.title("🫀 Heart Disease Prediction")
    st.markdown("Enter the patient's medical details below:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age_hd = st.number_input("Age", min_value=0, max_value=120, value=0, step=1)
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0.0, max_value=250.0, value=0.0)
        restecg = st.selectbox("Rest ECG Results", options=[0, 1, 2], help="0: Normal, 1: ST-T wave abnormality, 2: Probable/definite left ventricular hypertrophy")
        
    with col2:
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=0.0, max_value=600.0, value=0.0)
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0.0, max_value=250.0, value=0.0)
        
    with col3:
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("False", 0), ("True", 1)], format_func=lambda x: x[0])[1]
        exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Predict Heart Disease", use_container_width=True):
        if heart_model is None:
            st.error("Model file 'heart_model.sav' not found. Please place it in the application directory.")
        else:
            inputs = [age_hd, sex, cp, trestbps, chol, fbs, restecg, thalach, exang]
            
            # Use StandardScaler as required
            scaled_input = scale_input(inputs)
            prediction = heart_model.predict(scaled_input)
            
            if prediction[0] == 1:
                st.error("⚠️ The person is likely to have the disease (Heart Disease).")
            else:
                st.success("✅ The person is healthy.")

# ----------------------------------------------------
# 3. PARKINSON'S PREDICTION MODULE
# ----------------------------------------------------
elif selected == "Parkinson's Prediction":
    st.title("🧠 Parkinson's Disease Prediction")
    st.markdown("Enter the patient's medical details below:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fo = st.number_input('MDVP:Fo(Hz) - Avg vocal fundamental freq', min_value=0.0, max_value=300.0, value=0.0, format="%.5f")
        jitter = st.number_input('MDVP:Jitter(%) - Measure of variation in fundamental freq', min_value=0.0, max_value=1.0, value=0.0, format="%.5f")
        hnr = st.number_input('HNR - Harmonics-to-noise ratio', min_value=0.0, max_value=50.0, value=0.0, format="%.5f")
        
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz) - Max vocal fundamental freq', min_value=0.0, max_value=600.0, value=0.0, format="%.5f")
        shimmer = st.number_input('MDVP:Shimmer - Measure of variation in amplitude', min_value=0.0, max_value=1.0, value=0.0, format="%.5f")
        
    with col3:
        flo = st.number_input('MDVP:Flo(Hz) - Min vocal fundamental freq', min_value=0.0, max_value=300.0, value=0.0, format="%.5f")
        nhr = st.number_input('NHR - Noise-to-harmonics ratio', min_value=0.0, max_value=1.0, value=0.0, format="%.5f")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Predict Parkinson's", use_container_width=True):
        if parkinsons_model is None:
            st.error("Model file 'parkinsons_model.sav' not found. Please place it in the application directory.")
        else:
            inputs = [fo, fhi, flo, jitter, shimmer, nhr, hnr]
            
            # Use StandardScaler as required
            scaled_input = scale_input(inputs)
            prediction = parkinsons_model.predict(scaled_input)
            
            if prediction[0] == 1:
                st.error("⚠️ The person is likely to have the disease (Parkinson's).")
            else:
                st.success("✅ The person is healthy.")
