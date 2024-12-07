import streamlit as st
import joblib
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Loading the model and vectorizer
model = joblib.load('random_forest_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Initialize NLTK resources
stop_words = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()

# Function to preprocess the input text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation, and apply stemming
    tokens = [porter_stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]

    return ' '.join(tokens)


# App title and description
st.title("Medical Condition Prediction from Symptoms")
st.write(
    """
    ### Enter symptoms as a review text below:
    This app predicts medical conditions like **Depression**, **High Blood Pressure**, and **Diabetes, Type 2**.
    """
)

# Input text area for symptoms
symptoms_input = st.text_area("Symptoms (as in reviews):", height=150)

# Button to predict
if st.button("Predict Condition"):
    if symptoms_input:
        # Preprocess the symptoms
        preprocessed_symptoms = preprocess_text(symptoms_input)

        # Transform the text into TF-IDF features
        tfidf_features = tfidf_vectorizer.transform([preprocessed_symptoms])

        # Predict the condition
        prediction = model.predict(tfidf_features)

        # Decode the prediction
        predicted_condition = label_encoder.inverse_transform(prediction)

        # Display the result
        st.success(f"The predicted condition is: **{predicted_condition[0]}**")
    else:
        st.error("Please enter some symptoms to predict the condition.")

# Footer
st.write("Developed using RandomForest, TF-IDF, and Streamlit.")
