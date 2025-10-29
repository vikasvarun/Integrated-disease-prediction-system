import streamlit as st
import numpy as np
import joblib
import random

# Load models
adiposity_model = joblib.load(r"models/adiposity_random_forest_model.pkl")
parkinsons_model = joblib.load(r"models/parkinsons_model.pkl")
diabetes_model = joblib.load(r"models/diabetes_model.pkl")

# Symptom-specialist mapping
specialist_dict = {
    "headache": "Neurologist", "dizziness": "Neurologist", "pain in chest": "Cardiologist",
    "difficulty breathing": "Pulmonologist", "fever": "General Physician", "nausea": "Gastroenterologist",
    "abdominal pain": "Gastroenterologist", "joint pain": "Rheumatologist", "cough": "Pulmonologist",
    "skin rash": "Dermatologist", "swollen lymph nodes": "Oncologist", "sore throat": "ENT Specialist",
    "ear pain": "ENT Specialist", "eye redness": "Ophthalmologist", "burning urination": "Urologist",
    "frequent urination": "Urologist", "fatigue": "General Physician", "loss of appetite": "General Physician",
    "weight loss": "Endocrinologist", "palpitations": "Cardiologist", "vomiting": "Gastroenterologist",
    "back pain": "Orthopedist", "knee pain": "Orthopedist", "shoulder pain": "Orthopedist",
    "constipation": "Gastroenterologist", "diarrhea": "Gastroenterologist", "acne": "Dermatologist",
    "dry skin": "Dermatologist", "hair loss": "Dermatologist", "itching": "Dermatologist",
    "blurred vision": "Ophthalmologist", "double vision": "Ophthalmologist", "chest tightness": "Cardiologist",
    "shortness of breath": "Pulmonologist", "wheezing": "Pulmonologist", "nosebleed": "ENT Specialist",
    "night sweats": "General Physician", "cold hands": "General Physician", "swelling ankles": "Cardiologist",
    "numbness": "Neurologist", "tingling": "Neurologist", "muscle weakness": "Neurologist",
    "tremor": "Neurologist", "difficulty swallowing": "ENT Specialist", "hoarseness": "ENT Specialist",
    "bloating": "Gastroenterologist", "indigestion": "Gastroenterologist", "yellow skin": "Hepatologist",
    "yellow eyes": "Hepatologist", "anxiety": "Psychiatrist"
}

import streamlit as st

st.set_page_config(page_title="Disease Prediction System", layout="centered")

# Dummy models (just for demonstration)
models = {
    "Adiposity Prediction": "adiposity_model",
    "Parkinson's Prediction": "parkinsons_model",
    "Diabetes Prediction": "diabetes_model",
    "Symptom Checker": "symptom_checker"
}

# CSS Styling
st.markdown("""
    <style>
    .stButton>button {
        background-color: #ff7200;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 15px 30px;
        font-size: 14px;
        font-weight: bold;
        text-transform: uppercase;
        transition: background-color 0.3s ease;
        width: 220px;
        height: 60px;
    }
    .stButton>button:hover {
        background-color: white;
        color: #ff7200;
        border: 2px solid #ff7200;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)


st.title("ML-BASED DISEASE PREDICTION AND SYMPTOM ANALYZER")

# Layout: 2 rows x 2 columns
col1, col2 = st.columns(2)

with col1:
    if st.button("Adiposity Prediction"):
        st.session_state.page = "Adiposity Prediction"
    if st.button("Diabetes Prediction"):
        st.session_state.page = "Diabetes Prediction"

with col2:
    if st.button("Parkinson's Prediction"):
        st.session_state.page = "Parkinson's Prediction"
    if st.button("Symptom Checker"):
        st.session_state.page = "Symptom Checker"

# Show selected model
selected = st.session_state.get("page")

if selected == "Adiposity Prediction":
    st.write("Adiposity Prediction page loaded.")
elif selected == "Parkinson's Prediction":
    st.write("Parkinson's Prediction page loaded.")
elif selected == "Diabetes Prediction":
    st.write("Diabetes Prediction page loaded.")
elif selected == "Symptom Checker":
    st.write("Symptom Checker page loaded.")


# Adiposity Prediction Page
if selected == "Adiposity Prediction":
    st.header("🏋️‍♂️ Adiposity Prediction")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Age", 0, 100)
        weight = st.number_input("Weight (kg)", 20, 200)
        CH2O = st.number_input("CH2O (Water Intake)", 0.1, 10.0)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        height_unit = st.selectbox("Height Unit", ["Centimeters", "Feet"])
        if height_unit == "Centimeters":
            height = st.number_input("Height (cm)", 50, 300)
        else:
            height = st.number_input("Height (feet)", 1.5, 10.0) * 30.48

    with col3:
        FCVC = st.selectbox("FCVC (Veg Consumption)", ["Yes", "No"])
        NCP = st.number_input("NCP (Main Meals/Day)", 1, 5)
        family_history = st.selectbox("Family History", ["Yes", "No"])

    with col4:
        FAVC = st.selectbox("FAVC (Frequent Veg)", ["Yes", "No"])
        CALC = st.number_input("CALC (Calories)", 0, 10000)
        TUE = st.number_input("TUE (Energy Use)", 0, 10000)

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        MTRANS = st.selectbox("MTRANS (Transport)", ["Walking", "Car", "Bicycle", "Motorbike"])
        FAF = st.number_input("FAF (Activity/Week)", 0, 7)
    with col6:
        CAEC = st.selectbox("CAEC (Alcohol)", ["Yes", "No"])
    
    with col7:
        SMOKE = st.selectbox("SMOKE", ["Yes", "No"])
        
    with col8:
        SCC = st.selectbox("SCC (Physical Activity)", ["Yes", "No"])
      

    if st.button("Check Adiposity"):
        if all([age > 0, height > 0, weight > 0, NCP > 0, CH2O > 0, TUE >= 0, CALC >= 0, FAF >= 0]):
            if weight >= 120:
                st.success("🟢 Adiposity Detected")
            elif weight > 100 and MTRANS in ["Car", "Motorbike"]:
                st.success("🟢 Adiposity Detected")
            else:
                input_data = np.array([[age, height, weight, 1 if gender == 'Male' else 0,
                                        1 if family_history == 'Yes' else 0,
                                        1 if FAVC == 'Yes' else 0, 1 if FCVC == 'Yes' else 0, NCP,
                                        1 if CAEC == 'Yes' else 0, 1 if SMOKE == 'Yes' else 0, CH2O,
                                        1 if SCC == 'Yes' else 0, FAF, TUE, CALC,
                                        1 if MTRANS == 'Walking' else (2 if MTRANS == 'Car' else (3 if MTRANS == 'Bicycle' else 4))]])
                prediction = adiposity_model.predict(input_data)[0]
                st.success("🟢 Adiposity Detected" if prediction == 1 else "🔵 No Adiposity Detected")
        else:
            st.warning("⚠️ Please fill in all required fields.")
elif selected == "Parkinson's Prediction":
    st.header("🧠 Parkinson's Disease Prediction")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fo = st.number_input("MDVP:Fo(Hz)", value=random.uniform(100, 200))
        jitter_percent = st.number_input("MDVP:Jitter(%)", value=random.uniform(0, 1))
        rap = st.number_input("MDVP:RAP", value=random.uniform(0, 1))
        ddp = st.number_input("Jitter:DDP", value=random.uniform(0, 1))

    with col2:
        fhi = st.number_input("MDVP:Fhi(Hz)", value=random.uniform(100, 200))
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", value=random.uniform(0, 1))
        ppq = st.number_input("MDVP:PPQ", value=random.uniform(0, 1))
        shimmer = st.number_input("MDVP:Shimmer", value=random.uniform(0, 1))

    with col3:
        flo = st.number_input("MDVP:Flo(Hz)", value=random.uniform(100, 200))
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", value=random.uniform(0, 1))
        apq3 = st.number_input("Shimmer:APQ3", value=random.uniform(0, 1))
        apq5 = st.number_input("Shimmer:APQ5", value=random.uniform(0, 1))

    with col4:
        apq = st.number_input("MDVP:APQ", value=random.uniform(0, 1))
        dda = st.number_input("Shimmer:DDA", value=random.uniform(0, 1))
        nhr = st.number_input("NHR", value=random.uniform(0, 1))
        hnr = st.number_input("HNR", value=random.uniform(0, 1))

    col5, col6, col7 = st.columns(3)
    with col5:
        rpde = st.number_input("RPDE", value=random.uniform(0, 1))
        dfa = st.number_input("DFA", value=random.uniform(0, 1))

    with col6:
        spread1 = st.number_input("Spread1", value=random.uniform(0, 1))
        spread2 = st.number_input("Spread2", value=random.uniform(0, 1))

    with col7:
        d2 = st.number_input("D2", value=random.uniform(0, 1))
        ppe = st.number_input("PPE", value=random.uniform(0, 1))

    if st.button("Check Parkinson's Disease"):
        all_inputs = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                      shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                      rpde, dfa, spread1, spread2, d2, ppe]
        if all(all_inputs):
            input_data = np.array([all_inputs])
            prediction = parkinsons_model.predict(input_data)[0]
            st.success("🟢 Parkinson's Detected" if prediction == 1 else "🔵 No Parkinson's Detected")
        else:
            st.warning("⚠️ Please complete all fields.")


# Diabetes Prediction Page
elif selected == "Diabetes Prediction":
    st.header("🍩 Diabetes Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20)
        glucose = st.number_input("Glucose Level", 0, 200)
        blood_pressure = st.number_input("Blood Pressure", 0, 120)

    with col2:
        skin_thickness = st.number_input("Skin Thickness", 0, 100)
        insulin = st.number_input("Insulin", 0, 200)
        bmi = st.number_input("BMI", 0.0, 50.0)

    with col3:
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
        age = st.number_input("Age", 0, 120)

    if st.button("Check Diabetes"):
        if all([pregnancies >= 0, glucose >= 0, blood_pressure >= 0, skin_thickness >= 0, insulin >= 0, bmi >= 0, diabetes_pedigree >= 0, age >= 0]):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
            prediction = diabetes_model.predict(input_data)[0]
            st.success("🟢 Diabetes Detected" if prediction == 1 else "🔵 No Diabetes Detected")
        else:
            st.warning("⚠️ Please fill in all the required fields before checking for Diabetes.")

# Symptom Checker Page
elif selected == "Symptom Checker":
    st.header("🤔 Symptom Checker")
    st.write("""
        Select a symptom from the list, and we will guide you to the appropriate specialist.
        We will check based on common medical knowledge which specialist can help with your symptoms.
    """)
    symptom = st.selectbox("Symptom", list(specialist_dict.keys()))

    if st.button("Check Specialist"):
        if symptom:
            specialist = specialist_dict.get(symptom)
            st.success(f"The specialist for {symptom} is: {specialist}")
# Footer / Home Page Bottom Note
st.markdown("""
<hr style="border: 1px solid #ff7200;">
<div style='text-align: center; font-size: 16px; color: #333; padding-top: 20px;'>
    <strong>Note:</strong> This platform predicts Parkinson’s, diabetes, and adiposity using clinical data like blood pressure and heart rate. It offers real-time health insights, personalized recommendations, and leverages machine learning for accurate, accessible, and early disease detection, promoting preventive care.
</div>
""", unsafe_allow_html=True)
