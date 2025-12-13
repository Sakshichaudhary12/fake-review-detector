import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    filtered = []
    for word in text:
        if word.isalnum():
            filtered.append(word)

    filtered2 = []
    for word in filtered:
        if word not in stopwords.words('english') and word not in string.punctuation:
            filtered2.append(word)

    stemmed = []
    for word in filtered2:
        stemmed.append(ps.stem(word))

    return " ".join(stemmed)
