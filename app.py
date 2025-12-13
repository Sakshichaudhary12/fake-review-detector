import streamlit as st
import pickle
from utils import transform_text
import nltk

# Ensure stopwords/tokenizer available
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Fake Review Detector")

st.write("Enter a product review below and the model will classify it as Fake or Genuine.")

review = st.text_area("Write a review here:")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        transformed = transform_text(review)
        vector = tfidf.transform([transformed]).toarray()
        prediction = model.predict(vector)[0]

        if prediction == 0:
            st.error("Fake Review")
        else:
            st.success("Genuine Review")
