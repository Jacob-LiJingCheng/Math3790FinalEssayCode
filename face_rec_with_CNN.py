import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import time
import matplotlib.pyplot as plt

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

# Save the preprocessed data for future use
np.save('train_x.npy', train_x)
np.save('train_y.npy', train_y)
np.save('test_x.npy', test_x)
np.save('test_y.npy', test_y)

# Load the preprocessed data
train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')
test_x = np.load('test_x.npy')
test_y = np.load('test_y.npy')

# CNN implementation
def build_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(size, size, 1)),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(lb.classes_), activation='softmax')
    ])
    return model

cnn_model = build_cnn_model()
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class EpochHistory(Callback):
    def on_train_begin(self, logs=None):
        self.epoch_acc = []
        self.epoch_loss = []
        self.val_acc = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_acc.append(logs.get('accuracy'))
        self.epoch_loss.append(logs.get('loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.val_loss.append(logs.get('val_loss'))

epoch_history = EpochHistory()

# Early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# Record training time
start_time = time.time()
cnn_history = cnn_model.fit(train_x, train_y, batch_size=32, epochs=50, validation_data=(test_x, test_y), callbacks=[early_stopping, reduce_lr, epoch_history], verbose=1)
end_time = time.time()
cnn_training_time = end_time - start_time

# Record inference time
start_time = time.time()
cnn_loss, cnn_accuracy = cnn_model.evaluate(test_x, test_y, verbose=1)
end_time = time.time()
cnn_inference_time = end_time - start_time

print(f'CNN Test accuracy: {cnn_accuracy:.2f}')
print(f'CNN Training time: {cnn_training_time:.2f} seconds')
print(f'CNN Inference time: {cnn_inference_time:.2f} seconds')

# Save model
cnn_model.save('train_faces.keras')

# Plot training accuracy and loss over epochs
# Accuracy plot
plt.figure(figsize=(10, 5))
plt.plot(epoch_history.epoch_acc, label='CNN Training Accuracy')
plt.plot(epoch_history.val_acc, label='CNN Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('CNN Training and Validation Accuracy over Epochs')
plt.legend()
plt.savefig('cnn_training_validation_accuracy.png')

# Loss plot
plt.figure(figsize=(10, 5))
plt.plot(epoch_history.epoch_loss, label='CNN Training Loss')
plt.plot(epoch_history.val_loss, label='CNN Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('CNN Training and Validation Loss over Epochs')
plt.legend()
plt.savefig('cnn_training_validation_loss.png')
