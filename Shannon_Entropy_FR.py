import numpy as np
import cv2
import os

def calculate_entropy(image):
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    histogram = histogram / np.sum(histogram)  # Normalize histogram
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))  # Calculate entropy
    return entropy

data_dir = 'D:/att_faces'
size = 64

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
                        images.append(img)
                        labels.append(person_name)
    return np.array(images), np.array(labels)

images, labels = load_data(data_dir)

# Calculate entropy for each image
entropies = [calculate_entropy(image) for image in images]
print(f'Average entropy of the dataset: {np.mean(entropies):.2f}')
