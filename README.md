# AI-Powered-Personal-Health-Assistant
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk

# Sample dataset
data = {
    "symptoms": [
        "fever and cough", 
        "headache and nausea", 
        "sore throat and runny nose", 
        "chest pain and shortness of breath", 
        "fatigue and weight loss"
    ],
    "disease": [
        "flu",
        "migraine",
        "common cold",
        "heart disease",
        "diabetes"
    ]
}

df = pd.DataFrame(data)

# Build the ML pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train model
model.fit(df['symptoms'], df['disease'])

# Inference
def predict_disease(symptom_input):
    prediction = model.predict([symptom_input])[0]
    return prediction

# Example usage
user_input = input("Describe your symptoms: ")
predicted_disease = predict_disease(user_input)
print(f"Based on your symptoms, you might have: {predicted_disease}")
