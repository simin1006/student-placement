import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Load Trained Model
# -----------------------------
with open("placement_model.pkl", "rb") as file:
    model = pickle.load(file)

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Placement Prediction System")
st.write("Enter the student details below to predict placement status.")

st.markdown("---")

# -----------------------------
# User Input
# -----------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=30,
    value=21
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

degree = st.selectbox(
    "Degree",
    ["B.Sc", "B.Tech", "BCA", "M.Sc", "MBA"]
)

branch = st.selectbox(
    "Branch",
    ["CSE", "Civil", "Commerce", "Data Science", "EEE", "Finance", "IT", "ME"]
)

college_tier = st.selectbox(
    "College Tier",
    ["Tier-1", "Tier-2", "Tier-3"]
)

skills = st.number_input(
    "Skills Count",
    min_value=0,
    max_value=20,
    value=5
)

internships = st.number_input(
    "Internships",
    min_value=0,
    max_value=10,
    value=1
)

projects = st.number_input(
    "Projects",
    min_value=0,
    max_value=15,
    value=2
)

coding = st.selectbox(
    "Coding Level",
    ["Advanced", "Basic", "Intermediate"]
)

cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.5,
    step=0.01
)

package = st.number_input(
    "Package (LPA)",
    min_value=0.0,
    max_value=50.0,
    value=0.0,
    step=0.1
)
# -----------------------------
# Encoding Mapping
# -----------------------------

gender_map = {
    "Female": 0,
    "Male": 1
}

degree_map = {
    "B.Sc": 0,
    "B.Tech": 1,
    "BCA": 2,
    "M.Sc": 3,
    "MBA": 4
}

branch_map = {
    "CSE": 0,
    "Civil": 1,
    "Commerce": 2,
    "Data Science": 3,
    "EEE": 4,
    "Finance": 5,
    "IT": 6,
    "ME": 7
}

college_tier_map = {
    "Tier-1": 0,
    "Tier-2": 1,
    "Tier-3": 2
}

coding_level_map = {
    "Advanced": 0,
    "Basic": 1,
    "Intermediate": 2
}

# -----------------------------
# Prediction Button
# -----------------------------

if st.button("Predict Placement", use_container_width=True):

    input_data = np.array([[
        age,
        gender_map[gender],
        degree_map[degree],
        branch_map[branch],
        college_tier_map[college_tier],
        skills,
        internships,
        projects,
        coding_level_map[coding],
        cgpa,
        package
    ]])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:

        st.success("🎉 Congratulations!")
        st.success("The student is likely to be **PLACED**.")

        st.metric(
            label="Placement Probability",
            value=f"{probability[0][1] * 100:.2f}%"
        )

    else:

        st.error("❌ The student is likely to be **NOT PLACED**.")

        st.metric(
            label="Not Placed Probability",
            value=f"{probability[0][0] * 100:.2f}%"
        )

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")
st.markdown(
    "<center><h5>Developed using Streamlit & Machine Learning</h5></center>",
    unsafe_allow_html=True
)
