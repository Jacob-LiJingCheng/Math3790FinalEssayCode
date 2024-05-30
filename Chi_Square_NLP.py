import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

# Load the dataset
dataset_path = 'D:/IMDB Dataset.csv'
data = pd.read_csv(dataset_path)

# Convert labels to binary (0 for negative, 1 for positive)
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Convert text to feature vectors
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['review'])
y = data['sentiment']

# Perform chi-square test
chi2_scores, p_values = chi2(X, y)

# Select significant features
significant_features = np.where(p_values < 0.05)
print(f'Significant features count: {len(significant_features[0])}')

# Display top 10 significant features
top_features = np.argsort(chi2_scores)[-10:]
print("Top 10 significant features:")
for i in top_features:
    print(f'Feature: {vectorizer.get_feature_names_out()[i]}, Chi-square score: {chi2_scores[i]}, p-value: {p_values[i]}')
