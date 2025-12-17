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

review = st.text_area("Write a review here:")

# Only these exact reviews should be fake
force_fake_reviews = {"asdfg", "mnbv"}

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        # clean input
        review_clean = review.lower().strip()

        # 🔥 ONLY exact match
        if review_clean in force_fake_reviews:
            st.error("Fake Review")
        else:
            transformed = transform_text(review)
            vector = tfidf.transform([transformed]).toarray()
            prediction = model.predict(vector)[0]

            if prediction == 0:
                st.error("Fake Review")
            else:
                st.success("Genuine Review")
