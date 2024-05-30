import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import time
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data_dir = 'D:/att_faces'
size = 64

def getPaddingSize(img):
    """
    Calculate the padding size needed to make the image square.
    """
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
    """
    Load and preprocess the data from the given directory.
    """
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

# Add a channel dimension to the images
images = np.expand_dims(images, axis=-1)

# Convert labels to one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# Split the dataset into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(images, labels, test_size=0.2, random_state=42)

# Print dataset shapes
print(f'train_x shape: {train_x.shape}, train_y shape: {train_y.shape}')
print(f'test_x shape: {test_x.shape}, test_y shape: {test_y.shape}')

# Flatten the images to 1D vectors
n_samples, h, w, _ = train_x.shape
train_x_flat = train_x.reshape(n_samples, h * w)
test_x_flat = test_x.reshape(test_x.shape[0], h * w)

# Perform PCA for dimensionality reduction
n_components = 150
pca = PCA(n_components=n_components, whiten=True, random_state=42)
train_x_pca = pca.fit_transform(train_x_flat)
test_x_pca = pca.transform(test_x_flat)

# Train an SVM classifier
svc = SVC(kernel='linear', class_weight='balanced', verbose=True)

# Record training time
start_time = time.time()
svc.fit(train_x_pca, train_y.argmax(axis=1))
end_time = time.time()
pca_training_time = end_time - start_time

# Predict and record inference time
start_time = time.time()
pca_predictions = svc.predict(test_x_pca)
end_time = time.time()
pca_inference_time = end_time - start_time

# Evaluate the model
pca_accuracy = accuracy_score(test_y.argmax(axis=1), pca_predictions)
print(f'PCA + SVM Test accuracy: {pca_accuracy:.2f}')
print(f'PCA + SVM Training time: {pca_training_time:.2f} seconds')
print(f'PCA + SVM Inference time: {pca_inference_time:.2f} seconds')
