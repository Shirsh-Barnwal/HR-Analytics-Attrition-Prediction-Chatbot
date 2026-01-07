import pandas as pd
import os
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

#Loading Dataset

raw_path = "data/HR_Raw.csv"
df = pd.read_csv(raw_path)

print("Dataset loaded successfully!")
print("Shape: ",df.shape,"\n")
print("Columns: ",df.columns.tolist(),"\n")

#Creating Engagement Score Column

engagement_columns = [
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance"
]

for col in engagement_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column in dataset: {col} \n")
    
df["EngagementScore"] = df[engagement_columns].mean(axis=1)

#Create Engagement Level

def engagement_level(score):
    if score>=3.5: return "High"
    elif score>=2.5: return "Moderate"
    else: return "Low"
    
df["EngagementLevel"] = df["EngagementScore"].apply(engagement_level)

#Saving Processed Data

os.makedirs("processed", exist_ok=True)
processed_path = "processed/HR_processed.csv"
df.to_csv(processed_path, index=False)

print(f"Processed dataset saved at: {processed_path} \n")

#Load into SQLite Database

conn = sqlite3.connect("hr_analytics.db")
df.to_sql("employee_hr", conn, if_exists="replace", index=False)
conn.commit()

print("SQLite databse created: hr_analytics.db")
print("Table created: employee_hr")

#SQL Analytics Queries

cursor = conn.cursor()

print("\n--- Total Employees ---")
cursor.execute("SELECT COUNT(*) FROM employee_hr;")
print(cursor.fetchone()[0])

print("\n--- Attrition Distribution ---")
cursor.execute("SELECT Attrition, COUNT(*) FROM employee_hr GROUP BY Attrition;")
print(cursor.fetchall())

#Machine Learning Model

print("\n--- Training Machine Learning Model to Predict Attrition ---")
ml_df = df.copy()

label_target = LabelEncoder()
ml_df["Attrition"] = label_target.fit_transform(ml_df["Attrition"]) #Yes=1,No=0

feature_cols = [
    "Age",
    "MonthlyIncome",
    "TotalWorkingYears",
    "YearsAtCompany",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
    "EngagementScore"
]

X = ml_df[feature_cols]
y = ml_df["Attrition"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}")

print("\nClassification Report:")
print(classification_report(y_test,y_pred))

#Save the Model
os.makedirs("models",exist_ok=True)

MODEL_PATH = "models/attrition_model.pkl"
joblib.dump(model, MODEL_PATH)

print(f"\n Model saved successfully at: {MODEL_PATH}")
#saving label encoder too
ENCODER_PATH = "models/label_encoder.pkl"
joblib.dump(label_target, ENCODER_PATH)

print(f"Label Encoder saved successfully at: {ENCODER_PATH}")

conn.close()
print("\nStep complete: model training + saving finished successfully!")