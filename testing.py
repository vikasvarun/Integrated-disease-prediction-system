import streamlit as st
import numpy as np
import joblib

# Load models (replace with your actual file paths)
adiposity_model = joblib.load(r"C:\Users\sudeep\Desktop\m2\adiposity_random_forest_model.pkl")
parkinsons_model = joblib.load(r"C:\Users\sudeep\Desktop\m2\parkinsons_model.pkl")
diabetes_model = joblib.load(r"C:\Users\sudeep\Desktop\m2\diabetes_model.pkl")

# Adiposity Prediction Test
def test_adiposity(age, weight, height, gender, family_history, FAVC, FCVC, NCP, CH2O, MTRANS, FAF, CALC, TUE):
    input_data = np.array([[age, height, weight, 1 if gender == 'Male' else 0,
                            1 if family_history == 'Yes' else 0, 1 if FAVC == 'Yes' else 0,
                            1 if FCVC == 'Yes' else 0, NCP, 1 if MTRANS == 'Walking' else (2 if MTRANS == 'Car' else (3 if MTRANS == 'Bicycle' else 4)),
                            FAF, CALC, TUE, CH2O]])
    prediction = adiposity_model.predict(input_data)
    return prediction

# Parkinson's Prediction Test
def test_parkinsons(fo, jitter_percent, fhi, jitter_abs, ppq, shimmer, rpde, dfa, spread1, spread2):
    input_data = np.array([[fo, jitter_percent, fhi, jitter_abs, ppq, shimmer, rpde, dfa, spread1, spread2]])
    prediction = parkinsons_model.predict(input_data)
    return prediction

# Diabetes Prediction Test
def test_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    prediction = diabetes_model.predict(input_data)
    return prediction

# Streamlit Interface
st.title("ML-BASED DISEASE PREDICTION TESTING")

# Adiposity Test
st.header("Adiposity Prediction Test")
age = st.number_input("Age", 0, 100, 30)
weight = st.number_input("Weight (kg)", 20, 200, 70)
height = st.number_input("Height (cm)", 50, 300, 170)
gender = st.selectbox("Gender", ["Male", "Female"], index=0)
family_history = st.selectbox("Family History", ["Yes", "No"], index=0)
FAVC = st.selectbox("Frequent Veg", ["Yes", "No"], index=0)
FCVC = st.selectbox("Veg Consumption", ["Yes", "No"], index=0)
NCP = st.number_input("Meals per day", 1, 5, 3)
CH2O = st.number_input("Water Intake (L)", 0.1, 10.0, 2.0)
MTRANS = st.selectbox("Transport", ["Walking", "Car", "Bicycle", "Motorbike"], index=0)
FAF = st.number_input("Physical Activity (per week)", 0, 7, 3)
CALC = st.number_input("Calories Intake", 0, 10000, 2500)
TUE = st.number_input("Energy Usage", 0, 10000, 2000)

if st.button("Test Adiposity"):
    prediction = test_adiposity(age, weight, height, gender, family_history, FAVC, FCVC, NCP, CH2O, MTRANS, FAF, CALC, TUE)
    st.write(f"Adiposity Prediction: {'Adiposity Detected' if prediction == 1 else 'No Adiposity Detected'}")

# Parkinson's Test
st.header("Parkinson's Disease Prediction Test")
fo = st.number_input("MDVP:Fo (Hz)", 100, 200, 150)
jitter_percent = st.number_input("MDVP:Jitter(%)", 0.0, 1.0, 0.5)
fhi = st.number_input("MDVP:Fhi (Hz)", 100, 200, 150)
jitter_abs = st.number_input("MDVP:Jitter(Abs)", 0.0, 1.0, 0.5)
ppq = st.number_input("MDVP:PPQ", 0.0, 1.0, 0.5)
shimmer = st.number_input("MDVP:Shimmer", 0.0, 1.0, 0.5)
rpde = st.number_input("RPDE", 0.0, 1.0, 0.5)
dfa = st.number_input("DFA", 0.0, 1.0, 0.5)
spread1 = st.number_input("Spread1", 0.0, 1.0, 0.5)
spread2 = st.number_input("Spread2", 0.0, 1.0, 0.5)

if st.button("Test Parkinson's Disease"):
    prediction = test_parkinsons(fo, jitter_percent, fhi, jitter_abs, ppq, shimmer, rpde, dfa, spread1, spread2)
    st.write(f"Parkinson's Disease Prediction: {'Parkinson’s Detected' if prediction == 1 else 'No Parkinson’s Detected'}")

# Diabetes Test
st.header("Diabetes Prediction Test")
pregnancies = st.number_input("Pregnancies", 0, 20, 2)
glucose = st.number_input("Glucose Level", 0, 200, 100)
blood_pressure = st.number_input("Blood Pressure", 0, 120, 80)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 200, 90)
bmi = st.number_input("BMI", 0.0, 50.0, 25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 0, 120, 30)

if st.button("Test Diabetes"):
    prediction = test_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age)
    st.write(f"Diabetes Prediction: {'Diabetes Detected' if prediction == 1 else 'No Diabetes Detected'}")
