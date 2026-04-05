import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

# Load model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("💼 Employee Attrition Prediction")
st.caption("Predict whether an employee will leave or stay")

# Sidebar
st.sidebar.title("📌 About")
st.sidebar.info(
    "This ML app predicts employee attrition.\n\n"
    "👨‍💻 Developed by Chetan\n"
    "📊 Model: Random Forest"
)

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧾 Employee Details")
    satisfaction = st.slider("Satisfaction Level", 0.0, 1.0)
    evaluation = st.slider("Last Evaluation", 0.0, 1.0)
    projects = st.number_input("Number of Projects", 1, 10)
    hours = st.number_input("Average Monthly Hours", 50, 350)

with col2:
    st.subheader("📊 Work Information")
    time = st.number_input("Years in Company", 1, 10)
    accident = st.selectbox("Work Accident", ['Yes','No'])
    promotion = st.selectbox("Promotion in Last 5 Years", (1, 0))
    dept = st.selectbox("Department", 
        ['sales','technical','support','IT','hr','product_mng','marketing','RandD','accounting'])
    salary = st.selectbox("Salary", ['low','medium','high'])

# Encoding
accident = 1 if accident == 'Yes' else 0

dept_dict = {'sales':0,'technical':1,'support':2,'IT':3,'hr':4,'product_mng':5,'marketing':6,'RandD':7,'accounting':8}
salary_dict = {'low':0,'medium':1,'high':2}

dept = dept_dict[dept]
salary = salary_dict[salary]

st.divider()

# Prediction
if st.button("🔍 Predict"):
    input_data = np.array([[satisfaction, evaluation, projects, hours,
                            time, accident, promotion, dept, salary]])

    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)

    st.subheader("📢 Result")

    if prediction[0] == 1:
        st.error("🚨 Employee is likely to LEAVE")
    else:
        st.success("✅ Employee is likely to STAY")

    # Optional: Show input summary
    with st.expander("📄 View Input Summary"):
        st.write({
            "Satisfaction": satisfaction,
            "Evaluation": evaluation,
            "Projects": projects,
            "Hours": hours,
            "Years": time,
            "Accident": accident,
            "Promotion": promotion,
            "Department": dept,
            "Salary": salary
        })


