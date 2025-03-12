import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load dataset
data = pd.read_csv('/content/sample_data/Final_Augmented_dataset_Diseases_and_Symptoms.csv')
data = data.dropna()

# Prepare features & target
x = data.drop(columns=["diseases"])
y = data["diseases"]

# Encode target labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)  # Encode all diseases

# Split dataset
trainX, valX, trainY, valY = train_test_split(x, y_encoded, random_state=1)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(trainX, trainY)

# Function to predict disease from symptoms
def predict_disease(symptoms_dict):
    """
    Predicts the disease given a dictionary of symptom values.

    Args:
    symptoms_dict (dict): Dictionary where keys are symptom names and values are 0 or 1 (presence of symptom)

    Returns:
    str: Predicted disease name
    """
    # Convert symptoms dict to DataFrame
    symptoms_df = pd.DataFrame([symptoms_dict])

    # Ensure input has same columns as training data
    missing_cols = set(x.columns) - set(symptoms_df.columns)
    for col in missing_cols:
        symptoms_df[col] = 0  # Fill missing symptoms with 0

    # Reorder columns to match training data
    symptoms_df = symptoms_df[x.columns]

    # Predict probability scores for all diseases
    probabilities = model.predict_proba(symptoms_df)[0]

    # Get the indices of the top 3 most likely diseases
    top_indices = np.argsort(probabilities)[-3:][::-1]  # Sort in descending order

    # Convert indices to disease names
    top_diseases = encoder.inverse_transform(top_indices)
    top_probs = probabilities[top_indices]  # Get top 3 probabilities
     # Format the output
    predictions = {disease: round(prob * 100, 2) for disease, prob in zip(top_diseases, top_probs)}

    return predictions  # Returns a dictionary with diseases & confidence scores

# Test the function
preds = predict_disease({'fever': 1, "cough": 1, "headache": 0, "sore_throat": 1})
print("Predicted Disease:", preds)