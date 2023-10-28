# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 03:46:09 2023

@author: steli-garoz
"""

import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Path to IMDb reviews dataset file
dataset_path = 'imdb_dataset.csv'

data = pd.read_csv(dataset_path)

# Replace the string value with 0, 1
data['sentiment'] = data['sentiment'].map({'negative': 0, 'positive': 1})

# Use less data for development reasons
data = data.iloc[0:1500, :]
labels = data['sentiment'][:1500]

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

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the max_features as needed

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(data['lemmatized_text'].apply(lambda x: ' '.join(x)))

# You now have a TF-IDF matrix containing numerical representations of your text data
print(tfidf_matrix.shape)

# Assuming 'tfidf_matrix' is your TF-IDF feature matrix, and 'labels' is your sentiment labels (0 for negative, 1 for positive)
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, labels, test_size=0.2, random_state=42)

# Create a logistic regression model
logistic_regression_model = LogisticRegression()

# Train the model on the training data
logistic_regression_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = logistic_regression_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)

new_text = ["This is a shit movie... Don't worth it. I would prefer to eat bananas with onion. Disaster"]

# Transform the new text using the TF-IDF vectorizer
new_text_tfidf = tfidf_vectorizer.transform(new_text)

# Predict sentiment
predicted_sentiment = logistic_regression_model.predict(new_text_tfidf)

print("Predicted Sentiment:", "Positive" if predicted_sentiment[0] == 1 else "Negative")
