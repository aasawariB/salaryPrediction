
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_filename = 'random_forest_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Recreate LabelEncoders with the original classes
# These classes must match those used during training
label_encoders = {
    'Gender': LabelEncoder(),
    'Education Level': LabelEncoder(),
    'Job Title': LabelEncoder()
}

# Assign classes manually to avoid fitting on new data
label_encoders['Gender'].classes_ = ['Female', 'Male']
label_encoders['Education Level'].classes_ = ["Bachelor's", "Master's", 'PhD']

# Job Title has many classes, manually assign them or ensure consistency
# For simplicity, if a new job title appears, it will raise an error unless handled
# For demonstration, we use the original list of classes, ensuring it's robust
original_job_title_classes = [
    'Account Manager', 'Accountant', 'Administrative Assistant', 'Business Analyst', 'Business Development Manager',
    'Business Intelligence Analyst', 'CEO', 'Chief Data Officer', 'Chief Technology Officer', 'Content Marketing Manager', 
    'Copywriter', 'Customer Service Rep', 'Data Analyst', 'Data Scientist', 'DevOps Engineer', 'Digital Marketing Manager',
    'Director', 'Director of HR', 'Director of Marketing', 'Financial Analyst', 'Financial Manager', 'Full Stack Engineer',
    'Graphic Designer', 'HR Manager', 'IT Manager', 'Junior Developer', 'Junior Software Engineer', 'Marketing Analyst',
    'Marketing Coordinator', 'Marketing Manager', 'Operations Manager', 'Product Manager', 'Project Manager',
    'Recruiting Manager', 'Research Scientist', 'Sales Associate', 'Sales Director', 'Sales Manager',
    'Software Developer', 'Software Engineer', 'Software Engineer Manager', 'Social Media Manager', 'Technical Writer',
    'UX Designer', 'VP of Finance', 'VP of Operations', 'Web Developer', 'Junior Data Analyst', 'Senior Project Manager',
    'Senior HR Manager', 'Junior Web Developer', 'Staff Accountant', 'Research Analyst', 'Estimator', 'Business Analyst',
    'Financial Advisor', 'Digital Marketing Specialist', 'Creative Director', 'Help Desk Analyst', 'Public Relations Specialist',
    'Operations Analyst', 'Senior Software Engineer', 'Product Designer', 'Recruiter', 'Cloud Engineer', 'Senior Accountant',
    'Sales Representative', 'Human Resources Manager', 'HR Generalist', 'IT Support Specialist', 'Training Specialist',
    'Compliance Officer', 'Event Manager', 'Business Development Representative', 'Data Entry Clerk', 'Sales Executive',
    'Project Coordinator', 'Account Executive', 'Help Desk Technician', 'Systems Administrator', 'Network Engineer',
    'Database Administrator', 'UX Researcher', 'Social Media Specialist', 'Demand Generation Manager', 'Payroll Manager',
    'HR Coordinator', 'Content Creator', 'Technical Support Specialist', 'Quality Assurance Engineer', 'Marketing Specialist',
    'Business Systems Analyst', 'Supply Chain Manager', 'Frontend Developer', 'Backend Developer', 'UI/UX Designer',
    'Social Media Marketing Manager', 'Software Tester', 'Customer Success Manager', 'Network Administrator', 'Data Entry Specialist',
    'Senior Business Analyst', 'Senior Marketing Manager', 'Senior Data Scientist', 'Full Stack Web Developer', 'DevOps Specialist',
    'Database Developer', 'Data Engineer', 'Software QA Engineer', 'IT Support Manager', 'Technical Recruiter',
    'Senior Data Analyst', 'Sales Manager', 'HR Business Partner', 'Product Marketing Manager', 'Business Intelligence Analyst',
    'Senior Software Developer', 'Marketing Director', 'Senior Product Manager', 'Chief Marketing Officer', 'Chief Financial Officer',
    'Chief Technology Officer', 'Director of Sales', 'Director of Marketing', 'Director of Operations', 'Senior Account Manager',
    'Senior Marketing Analyst', 'Senior Financial Analyst', 'Principal Engineer', 'Staff Engineer', 'Associate Software Engineer',
    'Intern Software Engineer', 'Junior QA Engineer', 'Senior UX Designer', 'UI Designer', 'Copy Editor',
    'Content Manager', 'Marketing Intern', 'Sales Intern', 'HR Intern', 'Financial Intern', 'Business Development Intern',
    'Data Science Intern', 'Software Engineering Intern', 'UX Design Intern', 'Graphic Design Intern', 'Web Development Intern',
    'Project Management Intern', 'Customer Service Intern', 'IT Support Intern', 'Operations Intern', 'Product Management Intern',
    'Research Intern', 'Sales Associate Intern', 'Social Media Intern', 'Technical Writing Intern', 'Training Intern',
    'Compliance Intern', 'Event Planning Intern', 'Human Resources Intern', 'Marketing Research Analyst', 'Sales Operations Manager',
    'Senior HR Business Partner', 'Senior Marketing Specialist', 'Senior Sales Executive', 'Supply Chain Analyst', 'Talent Acquisition Specialist',
    'Technical Project Manager', 'VP of Sales', 'Customer Success Specialist', 'Cybersecurity Analyst', 'E-commerce Manager',
    'Executive Assistant', 'Operations Coordinator', 'Quality Assurance Analyst', 'Solutions Architect', 'Talent Management Specialist',
    'Technical Trainer', 'Training Manager', 'VP of Engineering', 'UX/UI Designer', 'Cloud Architect',
    'Data Science Manager', 'Director of Product Management', 'Director of Software Engineering', 'Enterprise Account Manager',
    'Financial Controller', 'Information Security Analyst', 'Lead Data Scientist', 'Lead Software Engineer', 'Marketing Communications Manager',
    'Principal Data Scientist', 'Principal Software Engineer', 'Product Owner', 'Sales Engineer', 'Senior Cloud Engineer',
    'Senior DevOps Engineer', 'Senior Network Engineer', 'Senior Systems Administrator', 'Software Engineering Manager', 'Solutions Engineer',
    'System Administrator', 'Technical Account Manager', 'Technical Product Manager', 'VP of Human Resources', 'VP of Marketing',
    'VP of Product Management', 'VP of Sales and Marketing', 'Head of Marketing', 'Head of Sales', 'HR Director',
    'Head of Product', 'Chief Operating Officer'
]
label_encoders['Job Title'].classes_ = original_job_title_classes


st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("Salary Prediction App")

st.write("Enter the details below to predict the salary.")

# User inputs
age = st.slider("Age", 18, 65, 30)

gender = st.selectbox("Gender", label_encoders['Gender'].classes_)

education_level = st.selectbox("Education Level", label_encoders['Education Level'].classes_)

years_of_experience = st.slider("Years of Experience", 0.0, 40.0, 5.0, 0.5)

# Job Title selection with an expander for better UX
with st.expander("Select Job Title (174 options)"):
    job_title = st.selectbox("", label_encoders['Job Title'].classes_)

# Predict button
if st.button("Predict Salary"):
    try:
        # Encode categorical features
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        education_encoded = label_encoders['Education Level'].transform([education_level])[0]
        job_title_encoded = label_encoders['Job Title'].transform([job_title])[0]

        # Create a DataFrame for the prediction
        input_data = pd.DataFrame([[age, gender_encoded, education_encoded, job_title_encoded, years_of_experience]],
                                  columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

        # Make prediction
        prediction = model.predict(input_data)[0]

        st.success(f"Predicted Salary: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please ensure all inputs are valid and try again.")

st.markdown("""
--- 
This app uses a Random Forest Regressor model trained on salary data.
""")
