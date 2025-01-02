import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
# Load the trained model
model = joblib.load('model.pkl')

# Function to preprocess and predict based on input data
def predict(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]  # Return the predicted class
def map_education(education_value):
    # Assuming 'None' -> 0, 'Bachelors' -> 1, 'Higher' -> 2, etc.
    education_mapping = {
        'None': 0,
        'Bachelors': 1,
        'Higher': 2,
        'Masters': 3,
    }
    return education_mapping.get(education_value, 0) 
def generate_pdf(input_data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.cell(200, 10, txt="Alzheimer's Disease Prediction Form", ln=True, align="C")
    pdf.ln(10)

    # Add form data to the PDF
    for key, value in input_data.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    result_text = "Presence of Alzheimer's disease" if prediction == 1 else "Absence of Alzheimer's disease"
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction Result: {result_text}", ln=True)    

    # Output PDF to a file-like object
    pdf_output = pdf.output(dest='S').encode('latin1') 
    return pdf_output

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        font-family: Arial, sans-serif;
    }
    .sidebar {
        background-color: #2c3e50;
        color: white;
    }
    .title {
        font-size: 32px;
        font-weight: bold;
        color: #3498db;
    }
    .form-container {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    .btn-submit {
        background-color: #3498db;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Streamlit App UI
st.sidebar.header("Navigation")
option = st.sidebar.radio("Choose an option:", ("Dashboard", "Fill the Form"))

if option == "Dashboard":
    st.markdown("<h1 class='title'>Alzheimer's Disease Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.write("Welcome to the Alzheimer's Disease Prediction Dashboard. Use the form to input patient data and predict the likelihood of Alzheimer's disease.")

elif option == "Fill the Form":
    st.markdown("<h2 class='title'>Please Fill in the Patient Information</h2>", unsafe_allow_html=True)
    
    with st.form(key='prediction_form'):
        # Input fields for each feature in the dataset
        age = st.number_input('Age', min_value=0, max_value=120, value=50)
        gender = st.selectbox('Gender', options=['Male', 'Female'])
        education = st.selectbox('Education', options=['None', 'Bachelors', 'Higher', 'Masters'])
        education_mapped = map_education(education)
        bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=25.0)
        smoking = st.selectbox('Smoking', options=['Yes', 'No'])
        alcohol = st.number_input('Alcohol Consumption (Rating 0 to 20)', min_value=0, max_value=20, value=0)
        physical_activity = st.number_input('Physical Activity(hours per week)', min_value=0, max_value=10, value=0)
        diet_quality = st.number_input('Diet Quality (Rating 0 to 10)', min_value=0, max_value=10, value=0)
        sleep_quality = st.number_input('Sleep Quality (Rating 0 to 20)', min_value=0, max_value=20, value=0)
        family_history = st.selectbox('Family History of Alzheimer’s', options=['Yes', 'No'])
        cardiovascular_disease = st.selectbox('Cardiovascular Disease', options=['Yes', 'No'])
        diabetes = st.selectbox('Diabetes', options=['Yes', 'No'])
        depression = st.selectbox('Depression', options=['Yes', 'No'])
        head_injury = st.selectbox('Head Injury', options=['Yes', 'No'])
        hypertension = st.selectbox('Hypertension', options=['Yes', 'No'])
        systolic_bp = st.number_input('Systolic Blood Pressure', min_value=50, max_value=200, value=120)
        diastolic_bp = st.number_input('Diastolic Blood Pressure', min_value=30, max_value=120, value=80)
        cholesterol_total = st.number_input('Total Cholesterol', min_value=100, max_value=300, value=200)
        cholesterol_ldl = st.number_input('LDL Cholesterol', min_value=50, max_value=200, value=100)
        cholesterol_hdl = st.number_input('HDL Cholesterol', min_value=30, max_value=100, value=50)
        cholesterol_triglycerides = st.number_input('Triglycerides', min_value=50, max_value=300, value=150)
        mmse = st.number_input('MMSE Score', min_value=0, max_value=30, value=25)
        memory_complaints = st.selectbox('Memory Complaints', options=['Yes', 'No'])
        behavioral_problems = st.selectbox('Behavioral Problems', options=['Yes', 'No'])
        confusion = st.selectbox('Confusion', options=['Yes', 'No'])
        disorientation = st.selectbox('Disorientation', options=['Yes', 'No'])
        personality_changes = st.selectbox('Personality Changes', options=['Yes', 'No'])
        difficulty_completing_tasks = st.selectbox('Difficulty Completing Tasks', options=['Yes', 'No'])
        forgetfulness = st.selectbox('Forgetfulness', options=['Yes', 'No'])

        # Submit button for form
        submit_button = st.form_submit_button(label='Predict')

        def map_yes_no(value):
    # Convert 'Yes' to 1, 'No' to 0
           return 1 if value == 'Yes' else 0

    # Process the form submission
    if submit_button:
        # Map the form inputs to match the dataset's feature names
        input_data = {
            'Age': age,
            'Gender': 0 if ['Gender'] == 'Male' else 1,  
            'EducationLevel': map_education(education),
            'BMI': bmi,
            'Smoking': map_yes_no(smoking),
            'AlcoholConsumption': alcohol,
            'PhysicalActivity': physical_activity,
            'DietQuality': diet_quality,
            'SleepQuality': sleep_quality,
            'FamilyHistoryAlzheimers': map_yes_no(family_history),
            'CardiovascularDisease': map_yes_no(cardiovascular_disease),
            'Diabetes': map_yes_no(diabetes),
            'Depression': map_yes_no(depression),
            'HeadInjury': map_yes_no(head_injury),
            'Hypertension': map_yes_no(hypertension),
            'SystolicBP': systolic_bp,
            'DiastolicBP': diastolic_bp,
            'CholesterolTotal': cholesterol_total,
            'CholesterolLDL': cholesterol_ldl,
            'CholesterolHDL': cholesterol_hdl,
            'CholesterolTriglycerides': cholesterol_triglycerides,
            'MMSE': mmse,
            'MemoryComplaints': map_yes_no(memory_complaints),
            'BehavioralProblems': map_yes_no(behavioral_problems),
            'Confusion': map_yes_no(confusion),
            'Disorientation': map_yes_no(disorientation),
            'PersonalityChanges': map_yes_no(personality_changes),
            'DifficultyCompletingTasks': map_yes_no(difficulty_completing_tasks),
            'Forgetfulness': map_yes_no(forgetfulness)
        }

        # Predict and display the result
        prediction = predict(input_data)
        if prediction == 1:
            st.write("The prediction indicates the presence of Alzheimer's disease.")
        else:
            st.write("The prediction indicates the absence of Alzheimer's disease.")

        pdf_output = generate_pdf(input_data)

            # Provide download button for the PDF
        st.download_button(
                label="Download the Form as PDF",
                data=pdf_output,
                file_name="alzheimers_disease_form.pdf",
                mime="application/pdf"  )
        

        








    

elif option == "About":
    st.write("This application predicts the likelihood of Alzheimer's disease based on various health and lifestyle factors. Fill out the form with the relevant information to receive a prediction.")

