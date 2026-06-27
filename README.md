# 📊 Employee Attrition Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

##  Live Demo

🔗 **Streamlit Application:**https://employee-attrition-mswrge5xos4hat7od2f38e.streamlit.app/

---

#  Project Overview

Employee attrition is a major concern for organizations because replacing skilled employees incurs significant recruitment, training, and productivity costs. This project develops a machine learning solution to predict whether an employee is likely to leave the company based on demographic, professional, and workplace-related features.

The project follows a complete Data Science lifecycle—from data preprocessing and exploratory data analysis (EDA) to model training, evaluation, and deployment through a user-friendly Streamlit web application.

---

#  Business Objective

The primary objective is to build a predictive model that enables Human Resource (HR) departments to:

- Identify employees at high risk of attrition.
- Improve employee retention strategies.
- Reduce hiring and training costs.
- Support data-driven workforce planning.
- Enable proactive employee engagement.

---

#  Dataset Information

| Attribute | Description |
|------------|-------------|
| Dataset | IBM HR Employee Attrition Dataset |
| Problem Type | Binary Classification |
| Target Variable | Attrition (Yes / No) |
| Records | 1470 Employees *(Update if different)* |
| Features | 35 *(Update if different)* |

---

#  Exploratory Data Analysis (EDA)

Exploratory Data Analysis was conducted to understand employee behavior, identify trends, detect outliers, and uncover relationships between variables.

## Key Insights

###  Attrition Distribution

- Approximately **84%** of employees stayed with the company.
- Around **16%** left the organization.
- The dataset exhibits class imbalance, which was considered during model evaluation.

###  Age Analysis

- Employees aged **25–35 years** showed the highest attrition rate.
- Attrition decreases as employee age increases.

###  Monthly Income

Employees with lower monthly income were significantly more likely to leave compared to higher-income employees.

### ✔ Overtime

Employees working overtime had substantially higher attrition rates, making overtime one of the strongest predictive features.

### ✔ Job Satisfaction

Lower job satisfaction was associated with increased employee turnover.

### ✔ Work-Life Balance

Poor work-life balance showed a strong relationship with employee attrition.

### ✔ Years at Company

Employees with fewer years at the company were more likely to leave.

### ✔ Marital Status

Single employees exhibited higher attrition compared to married employees.

---

# 🛠 Data Preprocessing

The following preprocessing techniques were applied:

- Missing value handling
- Duplicate removal
- Label Encoding
- One-Hot Encoding
- Feature Scaling
- Feature Selection
- Train-Test Split

---

#  Machine Learning Models

The following classification algorithms were implemented and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes
- XGBoost *(if used)*

---

#  Model Performance

| Model | Accuracy |
|---------|-----------|
| Logistic Regression | 77% |
| Decision Tree | 97% |
| Random Forest | 98% |
| KNN | 95% |




---

# 🏆 Best Model

**Model:** Random Forest Classifier 

Evaluation Metrics:

Accuracy : 98%





---

# Feature Importance

The model identified the following features as the most influential in predicting employee attrition:

- OverTime
- MonthlyIncome
- Age
- YearsAtCompany
- TotalWorkingYears
- JobSatisfaction
- EnvironmentSatisfaction
- StockOptionLevel
- DistanceFromHome
- JobRole

---

# Business Insights

The analysis suggests that employees are more likely to leave when they:

- Work overtime frequently.
- Receive lower monthly income.
- Have fewer years of experience in the organization.
- Experience low job satisfaction.
- Report poor work-life balance.
- Are younger employees early in their careers.

These insights can help HR departments implement targeted retention strategies and improve employee satisfaction.

---

#  Streamlit Web Application

The trained model has been deployed using **Streamlit Community Cloud**, providing an interactive and user-friendly interface.

### Application Features

- User-friendly dashboard
- Employee information input form
- Real-time attrition prediction
- Prediction confidence score *(if implemented)*
- Clean and responsive interface
- Fast model inference



# Technology Stack

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost *(if used)*
- Joblib

### Deployment

- Streamlit
- Streamlit Community Cloud

### Development Tools

- Jupyter Notebook
- VS Code
- Git
- GitHub

---

#  Project Structure

Employee-Attrition-Prediction/

│

├── data/

├── notebook/

├── models/

├── images/

├── app.py

├── prediction.py

├── requirements.txt

├── employee_attrition.csv

├── model.pkl

├── scaler.pkl

└── README.md

---

# Key Learning Outcomes

This project demonstrates practical experience in:

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Classification Algorithms
- Model Evaluation
- Hyperparameter Tuning
- Streamlit Application Development
- Model Deployment
- Git & GitHub Version Control
- End-to-End Machine Learning Workflow

---

#  Author

Chetan Telagaon

Aspiring Data Scientist | Machine Learning Engineer|Data Analyst

📧 Email: your-chetu.talagaon@gmail.com

🔗 LinkedIn:https://www.linkedin.com/in/chetan-telagaon

💻 GitHub: https:https://github.com/chetan041103

---

## If you found this project helpful, consider giving it a star on GitHub!
