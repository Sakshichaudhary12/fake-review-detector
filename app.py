import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle
from utils import transform_text

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Fake Review Detector")

review = st.text_area("Write a review here:")

# Sirf exact meaningless inputs
force_fake_words = {"asdfg", "mnbv"}

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        review_clean = review.lower().strip()

        # RULE 1: exact meaningless word
        if review_clean in force_fake_words:
            st.error("Fake Review")

        # RULE 2: bahut short gibberish
        elif len(review_clean.split()) < 2:
            st.error("Fake Review")

        else:
            transformed = transform_text(review)
            vector = tfidf.transform([transformed])

            # Safety check
            if vector.nnz == 0:
                st.error("Fake Review")
            else:
                prediction = model.predict(vector)[0]

                if prediction == 0:
                    st.error("Fake Review")
                else:
                    st.success("Genuine Review")
