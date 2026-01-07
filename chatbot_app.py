import os
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import altair as alt
from openai import OpenAI

#page config
st.set_page_config(
    page_title="HR Analytics - Power BI Style", layout="wide"
)

#API KEY
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not set.")
    st.stop()
client = OpenAI(api_key=api_key)

#model loading
model = joblib.load("models/attrition_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

#data loading
def get_df():
    conn = sqlite3.connect("hr_analytics.db")
    df = pd.read_sql_query("SELECT * FROM employee_hr;", conn)
    conn.close()
    return df
df = get_df()


#Theme Colors
YELLOW = "#F2C811"
BG = "#F4F6FA"                  # light background so everything is visible
CARD = "white"                  # white cards (Power BI light theme)
HEADER = "#0F172A"              # deep navy visible header

#CSS
st.markdown(f"""
<style>
body {{ background-color: {BG};}}
.block-container {{padding-top:0.5rem;}}

/*Header*/
.header {{
    background-color: {HEADER};
    color: white;
    padding: 14px 18px;
    border-radius: 10px;}}

.card {{ 
    background-color: {CARD};
    border-radius: 10px;
    padding: 14px;
    border-left: 6px solid {YELLOW};
    border:1px solid #E5E7EB;
}}

.kpi{{
    background-color: white;
    padding: 14px;
    border-radius: 10px;
    border-left: 6px solid {YELLOW};
    border:1px solid #E5E7EB;
}}

.sidebar {{
    background-color: white;
    padding: 14px;
    border-radius: 10px;
    border:1px solid #E5E7EB;
}}

.menu{{
    padding:8px 10px;
    border-radius:6px;
    margin-bottom:6px;
}}

.menu-active{{
    background-color: #FFF3B0;
    border-left: 6px solid {YELLOW};
}}
.stButton>button {{
    background-color: {YELLOW};
    color: black;
    border: none;
    height: 38px;
    border-radius: 6px;
    font-weight: 600;}}

</style>""", unsafe_allow_html=True    
)

#Header
st.markdown(f"""
<div class="header">
    <h2>HR Analytics Dashboard</h2>
    <small>Attrition - Engagement - Chatbot - Prediction</small>
</div>
""", unsafe_allow_html=True)

st.write("")

#Layout
sidebar,main = st.columns([0.2,0.8])

#Sidebar Menu
with sidebar:
    st.markdown("<div class='sidebar'>", unsafe_allow_html=True)
    
    st.subheader("HR Analytics")
    
    st.write("Dashboard Home")
    st.write("Attrition Risk")
    st.write("Engagement Analysis")
    st.write("Employee Directory")
    st.write("Chat Assistant")
    st.write("Settings")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
#Main Content
with main:
    #KPI Tiles
    k1, k2, k3 = st.columns(3)
    
    total_emp = len(df)
    attrition_rate = round((df[df['Attrition'] =="Yes"].shape[0]/total_emp)*100,2)
    avg_eng = round(df["EngagementScore"].mean(),2)
    
    with k1:
        st.markdown("<div class='kpi'>Total Employees</div>", unsafe_allow_html=True)
        st.subheader(total_emp)
    with k2:
        st.markdown("<div class='kpi'>Attrition Rate (%)</div>", unsafe_allow_html=True)
        st.subheader(f"{attrition_rate}%")
    with k3:
        st.markdown("<div class='kpi'>Average Engagement Score</div>", unsafe_allow_html=True)
        st.subheader(avg_eng)
        
    st.write("")
    
    #visualizations
    
    st.subheader("Visual Analytics")
    
    ch1, ch2 = st.columns(2)
    
    #attrition by department
    with ch1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        dept_attr = df.groupby(["Department","Attrition"]).size().reset_index(name="Count")
        
        chart1 = (
            alt.Chart(dept_attr).mark_bar().encode(x="Count", y="Department",color="Attrition")
        )
        st.altair_chart(chart1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    #engagement level distribution
    with ch2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        eng= df["EngagementLevel"].value_counts().reset_index()
        eng.columns = ['Level','Count']
        
        chart2 = (alt.Chart(eng).mark_arc(innerRadius=60).encode(theta='Count', color='Level', tooltip=['Level','Count']))

        st.altair_chart(chart2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.write("")
    
#Chatbot + Predictor + Profile
    c1, c2, c3 = st.columns(3)
    
    #Chatbot restored
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("HR Chat Assistant")
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        q = st.text_input("Ask your HR question:")
        
        if st.button("Ask Me"):
            if q.strip():
                st.session_state.chat_history.append(("User", q))
                
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                   messages=st.session_state.chat_history)
                
                ans = response.choices[0].message.content
                st.session_state.chat_history.append({
                    "role":"assistant",'content':ans})
            for m in st.session_state.chat_history[-5:]:
                st.write(f"**{m['role'].capitalize()}:** {m['content']}") 
                
            st.markdown("</div>", unsafe_allow_html=True)
        
        
        #Predictor restored
        with c2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Attrition Predictor")
            
            age = st.slider("Age",18,60,30)
            income = st.slider("Monthly Income",1000,20000,5000)
            tw = st.slider("Total Working Years",0,40,5)
            yc = st.slider("Years at Company",0,40,2)
            
            js = st.select_slider("Job Satisfaction", [1,2,3,4])
            es = st.select_slider("Environment Satisfaction", [1,2,3,4])
            rs = st.select_slider("Relationship Satisfaction", [1,2,3,4])
            wlb = st.select_slider("Work-Life Balance", [1,2,3,4])
            
            if st.button("Predict Risk"):
                eng = (js+es+rs+wlb)/4
                
                features = np.array([[age,income,tw,yc,js,es,rs,wlb,eng]])
                
                pred = model.predict(features)[0]
                prob = model.predict_proba(features)[0][1]*100
                label = label_encoder.inverse_transform([pred])[0]
                
                st.write(f"Prediction: {label}")
                st.write(f"Risk Probability: {prob:.1f}%")
                
            st.markdown("</div>", unsafe_allow_html=True)
            
            #Employee Profile restored
            
            with c3:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Employee Profile & Risk")

                emp_id = st.selectbox("Select Employee ID", df["EmployeeNumber"].tolist())
                emp = df[df["EmployeeNumber"]==emp_id].iloc[0]
                
                st.write(f"Department: **{emp['Department']}**")
                st.write(f"Role: **{emp['JobRole']}**")
                st.write(f"Engagement Level: **{emp['EngagementLevel']}**")
                
                features = np.array([[
                    emp['Age'],
                    emp['MonthlyIncome'],
                    emp['TotalWorkingYears'],
                    emp['YearsAtCompany'],
                    emp['JobSatisfaction'],
                    emp['EnvironmentSatisfaction'],
                    emp['RelationshipSatisfaction'],
                    emp['WorkLifeBalance'],
                    emp['EngagementScore']
                ]])
                
                pred = model.predict(features)[0]
                prob = model.predict_proba(features)[0][1]*100
                label = label_encoder.inverse_transform([pred])[0]
                
                st.write(f"Predicted Attrition: **{label}**")
                st.write(f"Attrition Probability: **{prob:.1f}%**")
                
                st.markdown("</div>", unsafe_allow_html=True)