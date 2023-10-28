# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 03:46:09 2023

@author: steli-garoz
"""

import spacy
import pandas as pd

# Path to IMDb reviews dataset file
dataset_path = 'imdb_dataset.csv'

data = pd.read_csv(dataset_path)

# Use less data for development reasons
data = data.iloc[0:10, :]

# Load the language model
nlp = spacy.load("en_core_web_sm")

# Tokenize the text and store it in a new column 'tokenized_text'
data['tokenized_text'] = None  # Create an empty column to store tokenized text

# Remove stopwords and create a list of non-stopwords for each review
for i, text in enumerate(data['review']):
    doc = nlp(text)
    non_stopwords = [token.text for token in doc if not token.is_stop]
    data.at[i, 'tokenized_text'] = non_stopwords

    # Print a visual counter
    print(f"Tokenizing review {i + 1} of {len(data)}")

# Now, the 'tokenized_text' column contains lists of tokens for each review.
print(data['tokenized_text'].head())

# Apply lemmatization to the tokenized text
data['lemmatized_text'] = data['tokenized_text'].apply(lambda tokens: [token.lemma_ for token in nlp(" ".join(tokens))])

# Now, the 'lemmatized_text' column contains lists of lemmatized tokens for each review.
print(data['lemmatized_text'].head())

