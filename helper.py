# We define our helper functions here with all the neccessary imports 
# Only major commands like the if else block and pickle imports in app.py


import pandas as pd
import re
import string
from nltk.corpus import stopwords
from symspellpy import SymSpell
import pandas as pd
import spacy
import regex as re
from nltk.stem.porter import PorterStemmer




# transpose the functions to accept a string instead of a series of strings







# Removing punctuation with a few exceptions
exclude = "".join(set(string.punctuation) - set("?")) + chr(8211) # en dash

# chat words
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

# Initialising a module then importing dictionary
sym_spell = SymSpell(max_dictionary_edit_distance=2)
sym_spell.load_dictionary("./en-80k.txt",
                          term_index=0, count_index=1)

# Tokenisation as list transformation
nlp = spacy.load('en_core_web_sm')


# For stemming 
Stemmer = PorterStemmer()




# The big boy of Text Preprocessing

def process_text(text):

    text_RemovePunc = remove_punc(text)

    text_RemoveUrls = removeUrls(text_RemovePunc)

    text_ChatConversion = chat_conversion(text_RemoveUrls)

    text_SpellCorrected = Spell_correction(text_ChatConversion)

    text_removeStopwords = remove_stopwords(text_SpellCorrected)
    
    text_Tokenisation = tokenization(text_removeStopwords)

    text_stemmed = Stemming(text_Tokenisation)

    return text_stemmed





def remove_punc(text):
    translation_mapping = str.maketrans("","",exclude)
    translation_mapping[ord("?")] = " ?"
    return text.translate(translation_mapping)








def removeUrls(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text) 








# Chat word treatment
def chat_conversion(text):
    new_text = []
    for w in text.split():
        if w.upper () in chat_words:
            new_text. append(chat_words[w.upper()])
        else:
            new_text. append (w)
    return " ". join(new_text)







def Spell_correction(text_entry):
    if "?" in text_entry:
        corrected_text = sym_spell.lookup_compound(text_entry, max_edit_distance=2)
        return corrected_text[0].term + " ?"
    else:
        corrected_text = sym_spell.lookup_compound(text_entry, max_edit_distance=2)
        return corrected_text[0].term
    






# Doing stopword removal at end
def remove_stopwords(text):
    new_text = []
    
    for word in text.split():
        if word in stopwords.words("english"):
            new_text.append('')
        else:
            new_text.append(word)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)






"""
Just segregating to keep the codes seperate
     based on their origin
"""





# Tokenisation - From string to list of strings
def tokenization(text_entry):
    # Check if TextEntry is a string
    if isinstance(text_entry, str):

        # Removing extra spaces
        StrippedText = re.sub(r'\s+', ' ', text_entry).strip()
        
        # Converting the text into a spaCy document
        spacy_doc = nlp(StrippedText)

        # Return the list of tokens
        return [token.text for token in spacy_doc]
    else:
        # Handling non-string input - might not need it now
        return ["NotStr"]
    
    # Showing unreachable here - still keeping it
    return["okay"]






# Stemming and TextPreProcessing go brrrr
def Stemming(ListEntry):
    return [Stemmer.stem(word) for word in ListEntry]
        


