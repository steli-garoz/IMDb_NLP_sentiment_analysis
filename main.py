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
train_data = data.iloc[0:1000, :].copy()
labels = train_data['sentiment']
val_data = data.iloc[1000:2000, :].copy()
val_labels = val_data['sentiment']

num_documents = len(val_data)
print("Number of documents in val_data:", num_documents)

# Load the language model
nlp = spacy.load("en_core_web_sm")

# Create lists to store tokenized and lemmatized text
train_data['tokenized_text'] = [None] * len(train_data)
train_data['lemmatized_text'] = [None] * len(train_data)
val_data['tokenized_text'] = [None] * len(val_data)
val_data['lemmatized_text'] = [None] * len(val_data)

# Tokenize, remove stopwords, lemmatize and create a list of non-stopwords for each review in train data
for i, text in enumerate(train_data['review']):
    doc = nlp(text)
    non_stopwords = [token.text for token in doc if not token.is_stop]
    train_data.at[i, 'tokenized_text'] = non_stopwords
    train_data.at[i, 'lemmatized_text'] = [token.lemma_ for token in doc]

    # Print a visual counter
    print(f"Tokenizing review {i + 1} of {len(train_data)}")

# Tokenize, remove stopwords, lemmatize and create a list of non-stopwords for each review in test data
for i, text in enumerate(val_data['review']):
    doc = nlp(text)
    non_stopwords = [token.text for token in doc if not token.is_stop]
    val_data.at[1000+i, 'tokenized_text'] = non_stopwords
    val_data.at[1000+i, 'lemmatized_text'] = [token.lemma_ for token in doc]

    # Print a visual counter
    print(f"Tokenizing test review {i + 1} of {len(val_data)}")

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the max_features as needed

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['lemmatized_text'].apply(lambda x: ' '.join(x)))
tfidf_matrix_test = tfidf_vectorizer.transform(val_data['lemmatized_text'].apply(lambda x: ' '.join(x)))

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

# Make predictions on the validation data
val_predictions = logistic_regression_model.predict(tfidf_matrix_test)

# Calculate evaluation metrics for the validation data
val_accuracy = accuracy_score(val_labels, val_predictions)
val_precision = precision_score(val_labels, val_predictions)
val_recall = recall_score(val_labels, val_predictions)
val_f1 = f1_score(val_labels, val_predictions)
val_conf_matrix = confusion_matrix(val_labels, val_predictions)

print("Validation Accuracy:", val_accuracy)
print("Validation Precision:", val_precision)
print("Validation Recall:", val_recall)
print("Validation F1 Score:", val_f1)
print("Validation Confusion Matrix:")
print(val_conf_matrix)
