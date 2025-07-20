
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
df = pd.read_csv(r"C:/Unified Internship/Project 2/dataset.csv")

# Step 2: Fill missing values
df.fillna("none", inplace=True)

# Step 3: Get all unique symptoms
symptom_cols = [col for col in df.columns if "Symptom" in col]
all_symptoms = set()

for col in symptom_cols:
    all_symptoms.update(df[col].unique())

# Create mapping
symptom_to_int = {symptom: idx for idx, symptom in enumerate(sorted(all_symptoms))}
int_to_symptom = {v: k for k, v in symptom_to_int.items()}

# Encode symptoms into integers
for col in symptom_cols:
    df[col] = df[col].map(symptom_to_int)

# Encode disease labels
from sklearn.preprocessing import LabelEncoder
disease_le = LabelEncoder()
df['Disease'] = disease_le.fit_transform(df['Disease'])

X = df[symptom_cols]
y = df['Disease']

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

def predict_disease(symptoms_input):
    # Preprocess: pad or trim to expected input length
    symptoms_input = symptoms_input[:len(symptom_cols)] + ["none"] * (len(symptom_cols) - len(symptoms_input))
    
    try:
        encoded_input = [symptom_to_int[symptom] for symptom in symptoms_input]
    except KeyError as e:
        print(f"❌ Symptom '{e.args[0]}' not in symptom dictionary.")
        return

    prediction = clf.predict([encoded_input])[0]
    disease_name = disease_le.inverse_transform([prediction])[0]
    print(f"⚠️ Predicted Disease: {disease_name}")

# Example usage
predict_disease(["fever", "cough", "headache"])

