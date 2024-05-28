#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 19:36:19 2024

@author: ayush
"""

import pandas as pd
import re
import string
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from symspellpy import SymSpell
from nltk.stem import PorterStemmer
import pickle





# Load TF-IDF Vectorizer
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Load Random Forest Classifier
random_forest_classifier = pickle.load(open('rf_tfidf.pkl', 'rb'))

# Load stopwords
stop_words = set(stopwords.words("english"))

# Initialize Stemmer
stemmer = PorterStemmer()

# Initialize SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2)
sym_spell.load_dictionary("/Users/ayush/Documents/Stuff/ML Boi/Dictionary/en-80k.txt", term_index=0, count_index=1)



def preprocess_text(text):
    
    # Remove punctuation
    exclude = "".join(set(string.punctuation) - set("?")) + chr(8211)  # en dash
    translation_mapping = str.maketrans("", "", exclude)
    translation_mapping[ord("?")] = " ?"
    text = text.translate(translation_mapping)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Chat word treatment
    chat_words = {
        'lol': 'laugh out loud',
        'lmao': 'laughing my ass off',
        'gn': 'good night',
        'brb': 'be right back',
        'btw': 'by the way',
        'ill': 'i will',
        'aka': 'that is',
        'omg': "oh my god"
    }
    text = " ".join([chat_words.get(word.upper(), word) for word in text.split()])
    
    # Spell correction
    if "?" in text:
        # If "?" is present, keep it and correct the rest of the text
        corrected_text = sym_spell.lookup_compound(text, max_edit_distance=2)
        text = corrected_text[0].term + " ?"
    else:
        # If "?" is not present, perform regular spell correction
        corrected_text = sym_spell.lookup_compound(text, max_edit_distance=2)
        text = corrected_text[0].term
        
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])

    # Tokenization
    text = text.split()

    # Stemming
    text = [stemmer.stem(word) for word in text]
    
    return " ".join(text)

def predict_category(text):
    
    # Preprocess text
    text = preprocess_text(text)
    
    # Convert text to TF-IDF features
    tfidf_features = tfidf_vectorizer.transform(text)
    
    # Predict category
    category = random_forest_classifier.predict(tfidf_features)
    
    return category[0]

# Example usage
text = "You should do enquiry, changes, information, advice, comment"
preprocess_text(text)
#predicted_category = predict_category(text)
#print("Predicted category:", predicted_category)