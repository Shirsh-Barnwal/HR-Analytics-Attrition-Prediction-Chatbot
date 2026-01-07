import streamlit as st
import pandas as pd
import sqlite3
import joblib
import numpy as np

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")

st.title("HR Analytics - Employee Engagement and Attrition Dashboard")
st.write("Data Source: SQLite database (hr_analytics.db)")

#Loading Dataset from SQLite database

@st.cache_data
def load_data():
    conn = sqlite3.connect("hr_analytics.db")
    df = pd.read_sql_query("SELECT * FROM employee_hr;", conn)
    conn.close()
    return df

df=load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

#KPI Metrics

st.subheader("Key HR Metrics")

total_emp = len(df)
attrition_rate = (df[df['Attrition'] =="Yes"].shape[0]/total_emp)*100
avg_engagement = df["EngagementScore"]

col1,col2,col3 = st.columns(3)

with col1:
    st.metric("Total Employees", total_emp)
with col2:
    st.metric("Attrition Rate (%)", round(attrition_rate, 2))
with col3:
    st.metric("Average Engagement Score", round(avg_engagement.mean(), 2))
    
#Visualizations

st.subheader("Engagement Level Distribution")
st.bar_chart(df['EngagementLevel'].value_counts())

st.subheader("Attrition Distribution")
st.bar_chart(df['Attrition'].value_counts())

#Loading our ML model

model = joblib.load("models/attrition_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

st.subheader("Attrition Prediction Tool")
st.write("Enter employee details to predict attrition risk")

#Prediction Input Form

with st.form("prediction_form"):
    
    age = st.slider("Age",18,60,30)
    income = st.slider("Monthly Income",1000,20000,5000)
    total_work_years = st.slider("Total Working Years",0,40,5)
    years_at_company = st.slider("Years at Company",0,40,2)
    
    job_sat = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
    env_sat = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
    rel_sat = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3) 
    wlb = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
    
    submitted = st.form_submit_button("Predict Attrition")
    
    if submitted:
        
        engagement_score = (job_sat + env_sat + rel_sat + wlb)/4
        
        features = np.array([[
            age,
            income,
            total_work_years,
            years_at_company,
            job_sat,
            env_sat,
            rel_sat,
            wlb,
            engagement_score
        ]])
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] *100
        
        attrition_label = label_encoder.inverse_transform([prediction])[0]
        
        st.write("### Prediction Result")
        st.write(f"Predicted Attrition: **{attrition_label}**")
        st.write(f"Attrition Probability: **{probability:.2f}%**")