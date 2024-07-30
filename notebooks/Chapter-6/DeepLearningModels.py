import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Input
from tensorflow.keras.metrics import Precision, Recall, AUC

def build_rnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, return_sequences=True))
    model.add(GRU(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', Precision(), Recall(), AUC()])
    return model

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', Precision(), Recall(), AUC()])
    return model

def train_rnn_model(data, labels):
    input_shape = (data.shape[1], data.shape[2])
    model = build_rnn_model(input_shape)
    history = model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.2)
    return model, history

def train_cnn_model(data, labels):
    input_shape = (data.shape[1], data.shape[2], data.shape[3])
    model = build_cnn_model(input_shape)
    history = model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.2)
    return model, history
