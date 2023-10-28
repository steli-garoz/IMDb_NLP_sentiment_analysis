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

# Use only part of the data during development in order to work faster
data = data.iloc[0:1000, :]

# Load the language model
nlp = spacy.load("en_core_web_sm")

# Tokenize the text and store it in a new column 'tokenized_text'
data['tokenized_text'] = None  # Create an empty column to store tokenized text

# Add a counter to track progress
total_reviews = len(data)
for i, text in enumerate(data['review']):
    tokenized_text = [token.text for token in nlp(text)]
    data.at[i, 'tokenized_text'] = tokenized_text

    # Print a visual counter
    print(f"Tokenizing review {i + 1} of {total_reviews}")

# Now, the 'tokenized_text' column contains lists of tokens for each review.
print(data['tokenized_text'].head())

