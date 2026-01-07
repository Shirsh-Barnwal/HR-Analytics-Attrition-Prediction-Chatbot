import os
import sqlite3
import numpy as np
import joblib
from openai import OpenAI

#Loading API Key

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OpenAI API key not set in environment variables.")

client = OpenAI(api_key)

#SQLite Connection

DB_PATH = "hr_analytics.db"

def run_query(query):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return rows

#Loading ML Model

MODEL_PATH = "models/attrition_model.pkl"
model = joblib.load(MODEL_PATH)

ENCODER_PATH = "models/label_encoder.pkl"
label_encoder = joblib.load(ENCODER_PATH)

#Prediction Function

def predict_attrition():
    print("\nEnter Employee Details to Predict Attrition Risk:")
    
    age = int(input("Age: "))
    income = int(input("Monthly Income: "))
    total_work_years = int(input("Total Working Years: "))
    years_at_company = int(input("Years at Company: "))
    
    job_sat = int(input("Job Satisfaction (1-4): "))
    env_sat = int(input("Environment Satisfaction (1-4): "))
    rel_sat = int(input("Relationship Satisfaction (1-4): "))
    wlb = int(input("Work Life Balance (1-4): "))
    
    engagement_score = (job_sat + env_sat + rel_sat + wlb) / 4
    
    features = np.array([[age, income, total_work_years, years_at_company,job_sat, env_sat, rel_sat, wlb, engagement_score]])
    
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1] * 100
    label = label_encoder.inverse_transform([pred])[0]
    
    print("\n------------------------------")
    print(f"Predicted Attrition: {label}")
    print(f"Attrition Probability: {prob:.2f}%")
    print("------------------------------\n")
    
    #Chat Loop
    print("HR Analytics ChatGPT + SQL + ML Chatbot")
    print("Type 'predict' for ML attrition prediction")
    print("Type 'exit' to quit the chatbot")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot session ended.")
            break
        
        if user_input.lower() == "predict":
            predict_attrition()
            continue
        
        #Prompt for ChatGPT
        prompt = f"""
        You are an HR analytics assistant with SQL access to a table named employee_hr.

    Understand the user question and if needed produce ONLY a valid SQL query
    using fields such as:
    Age, Department, JobRole, Attrition, MonthlyIncome,
    EngagementScore, EngagementLevel.

    If SQL is not appropriate, answer normally in English.

    User question: {user_input}
    """
    
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"user","content":prompt}])
        
        ai_text = response.choices[0].message.content
        print("\nChatGPT:", ai_text)
        
        #Execute SQL if present
        if "SELECT" in ai_text.upper():
            try:
                result = run_query(ai_text)
                print("\nSQL Result:")
                for row in result:
                    print(row)
            except Exception as e:
                print("Error executing SQL:", e)