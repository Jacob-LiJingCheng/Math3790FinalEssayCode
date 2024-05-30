import numpy as np
import pandas as pd
from collections import Counter

def calculate_entropy(text):
    counter = Counter(text.split())
    total = len(text.split())
    entropy = -sum((count / total) * np.log2(count / total + 1e-7) for count in counter.values())
    return entropy

# Load the dataset
dataset_path = 'D:/IMDB Dataset.csv'
data = pd.read_csv(dataset_path)

# Calculate entropy for each review
entropies = [calculate_entropy(review) for review in data['review']]
print(f'Average entropy of the dataset: {np.mean(entropies):.2f}')
