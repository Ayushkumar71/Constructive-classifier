
import helper
import pickle
import streamlit as st


# Load TF-IDF Vectorizer
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Load Random Forest Classifier
random_forest_classifier = pickle.load(open('rf_tfidf.pkl', 'rb'))


def predict_category(text):
    ptext = helper.process_text(text)
    # Convert text to TF-IDF features
    tfidf_features = tfidf_vectorizer.transform(ptext)
    # Predict category
    category = random_forest_classifier.predict(tfidf_features)

    return category[0]





st.header('Constructiveness classifier')




Itext = st.text_input('Enter your comment')

if st.button('Find'):
    pred = predict_category(Itext)

    if pred == 1:
        print("Constructive")
        st.header('Constructive')

    elif pred == 0:
        print("Somewhat constructive")
        st.header('Somewhat constructive')

    elif pred == -1:
        print("Not constructive")
        st.header('Not constructive')
