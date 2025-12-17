import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle
from utils import transform_text

# Load model and vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Fake Review Detector")

st.write("Enter a product review below and the model will classify it as Fake or Genuine.")

review = st.text_area("Write a review here:")

# Words that should always be marked as fake
force_fake_words = ["asdfg", "mnbv"]

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        # Convert to lowercase for matching
        review_lower = review.lower().strip()

        # Rule-based override
        if review_lower in force_fake_words:
            st.error("Fake Review")
        else:
            transformed = transform_text(review)
            vector = tfidf.transform([transformed]).toarray()
            prediction = model.predict(vector)[0]

            if prediction == 0:
                st.error("Fake Review")
            else:
                st.success("Genuine Review")
