import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import time

# Load the dataset
dataset_path = 'D:/IMDB Dataset.csv'
data = pd.read_csv(dataset_path)

# Preprocess the data
X = data['review']
y = data['sentiment'].map({'positive': 1, 'negative': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to feature vectors
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train the Naive Bayes model
nb_model = MultinomialNB()

start_time = time.time()
nb_model.fit(X_train_counts, y_train)
training_time = time.time() - start_time

# Predict on the test set
start_time = time.time()
y_pred = nb_model.predict(X_test_counts)
inference_time = time.time() - start_time

# Evaluate the model
nb_accuracy = accuracy_score(y_test, y_pred)
print(f'Naive Bayes Test Accuracy: {nb_accuracy:.2f}')
print(f'Training time: {training_time:.2f} seconds')
print(f'Inference time: {inference_time:.2f} seconds')
