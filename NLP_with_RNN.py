import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = 'D:/IMDB Dataset.csv'
data = pd.read_csv(dataset_path)

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Extract features and labels
train_texts = train_data['review'].values
train_labels = train_data['sentiment'].values
test_texts = test_data['review'].values
test_labels = test_data['sentiment'].values

# Convert sentiment labels to binary values
train_labels = np.where(train_labels == 'positive', 1, 0)
test_labels = np.where(test_labels == 'positive', 1, 0)

# Tokenize the text data
max_words = 20000
max_len = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Pad the sequences
train_x = pad_sequences(train_sequences, maxlen=max_len)
test_x = pad_sequences(test_sequences, maxlen=max_len)

# One-hot encode the labels
lb = LabelBinarizer()
train_y = lb.fit_transform(train_labels)
test_y = lb.transform(test_labels)

# Print dataset shapes
print(f'train_x shape: {train_x.shape}, train_y shape: {train_y.shape}')
print(f'test_x shape: {test_x.shape}, test_y shape: {test_y.shape}')

# Save the preprocessed data for future use
np.save('train_x_imdb.npy', train_x)
np.save('train_y_imdb.npy', train_y)
np.save('test_x_imdb.npy', test_x)
np.save('test_y_imdb.npy', test_y)

# Load the preprocessed data
train_x = np.load('train_x_imdb.npy')
train_y = np.load('train_y_imdb.npy')
test_x = np.load('test_x_imdb.npy')
test_y = np.load('test_y_imdb.npy')

# RNN implementation with LSTM
def build_rnn_model():
    model = models.Sequential([
        layers.Embedding(max_words, 128, input_length=max_len),
        layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

rnn_model = build_rnn_model()
rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)

# Record training time
start_time = time.time()
rnn_history = rnn_model.fit(train_x, train_y, batch_size=32, epochs=20, validation_data=(test_x, test_y), callbacks=[early_stopping, reduce_lr, epoch_history])
end_time = time.time()
rnn_training_time = end_time - start_time

# Record inference time
start_time = time.time()
rnn_loss, rnn_accuracy = rnn_model.evaluate(test_x, test_y)
end_time = time.time()
rnn_inference_time = end_time - start_time

print(f'RNN Test accuracy: {rnn_accuracy:.2f}')
print(f'RNN Training time: {rnn_training_time:.2f} seconds')
print(f'RNN Inference time: {rnn_inference_time:.2f} seconds')

# Save model
rnn_model.save('train_rnn_imdb.keras')

# Plot RNN training and validation accuracy over epochs
plt.figure(figsize=(10, 5))
plt.plot(epoch_history.epoch_acc, label='RNN Training Accuracy')
plt.plot(epoch_history.val_acc, label='RNN Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('RNN Training and Validation Accuracy over Epochs')
plt.legend()
plt.savefig('rnn_training_validation_accuracy_imdb.png')

# Plot RNN training and validation loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(epoch_history.epoch_loss, label='RNN Training Loss')
plt.plot(epoch_history.val_loss, label='RNN Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('RNN Training and Validation Loss over Epochs')
plt.legend()
plt.savefig('rnn_training_validation_loss_imdb.png')
