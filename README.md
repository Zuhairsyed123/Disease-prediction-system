#  AI-Based Multi-Disease Prediction System

## Overview
This project is a machine learning-based healthcare prediction system that can detect multiple diseases, including Diabetes, Heart Disease, and Parkinson’s Disease.

The system integrates multiple trained models into a single platform and provides predictions based on user input through an interactive web interface built using Streamlit.

It is designed as a clinical decision support tool to assist in early-stage disease detection using data-driven insights.

---

##  Features
- Predicts multiple diseases in a single system
- Separate input modules for each disease
- Interactive user interface using Streamlit
- Real-time predictions based on user inputs
- Clean and user-friendly design

---

##  Diseases Covered

### 🔹 Diabetes Prediction
Predicts diabetes based on health parameters such as:
- Glucose level
- Blood pressure
- BMI
- Insulin
- Age

---

### 🔹 Heart Disease Prediction
Predicts heart disease risk using:
- Age
- Chest pain type
- Blood pressure
- Cholesterol
- ECG results
- Maximum heart rate

---

### 🔹 Parkinson’s Disease Prediction
Predicts Parkinson’s using biomedical voice features:
- Jitter
- Shimmer
- NHR (Noise-to-Harmonics Ratio)
- HNR (Harmonics-to-Noise Ratio)

---

##  Tech Stack
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Streamlit  
- Pickle  

---

##  Model Details
- Algorithm Used: Support Vector Machine (SVM)
- Feature Scaling: StandardScaler
- Separate models trained for each disease
- Models saved using Pickle for efficient reuse

---

##  Workflow
1. Data Collection  
2. Data Preprocessing  
3. Feature Scaling  
4. Model Training (SVM)  
5. Model Saving using Pickle  
6. UI Development using Streamlit  
7. Real-time Prediction  

---

## ▶️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
streamlit run app.py
