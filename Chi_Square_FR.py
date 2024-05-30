import os
import numpy as np
import cv2
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelBinarizer

# Data loading and preprocessing
data_dir = 'D:/att_faces'
size = 64

def getPaddingSize(img):
    h, w = img.shape[:2]
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    return top, bottom, left, right

def load_data(data_dir, h=size, w=size):
    images = []
    labels = []
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.endswith('.pgm'):
                    filepath = os.path.join(person_dir, filename)
                    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        top, bottom, left, right = getPaddingSize(img)
                        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        img = cv2.resize(img, (h, w))
                        images.append(img)
                        labels.append(person_name)
    return np.array(images), np.array(labels)

images, labels = load_data(data_dir)

# Convert labels to numerical format
lb = LabelBinarizer()
labels = lb.fit_transform(labels).argmax(axis=1)  # Convert one-hot encoding back to single label

# Simulate the presence of certain facial features as binary variables (0 or 1)
# In practice, these should be real features extracted from the images
np.random.seed(42)  # For reproducibility
features = np.random.randint(0, 2, size=(len(labels), 3))  # Simulated features
feature_names = ['Feature1', 'Feature2', 'Feature3']

# Create a DataFrame with labels and features
df = pd.DataFrame(features, columns=feature_names)
df['Label'] = labels

# Perform chi-square test for each feature
for feature in feature_names:
    contingency_table = pd.crosstab(df[feature], df['Label'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f'Chi-square test for {feature}:')
    print(f'Chi-square statistic: {chi2}')
    print(f'p-value: {p}')
    print(f'Degrees of freedom: {dof}')
    print('Expected frequencies:')
    print(expected)
    print('---')
