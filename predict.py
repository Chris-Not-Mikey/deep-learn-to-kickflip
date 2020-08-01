import csv
import cv2
import tensorflow as tf
import keras
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from train import read_video






if __name__ == "__main__":
    
    file = "./videos/validate/validation_video.mp4"
    frames, old_count = read_video(file)
    X_test = np.array(frames)
    X = X_test[0:10]
    cnn_filename = "./cnn_models"
    model = tf.keras.models.load_model(cnn_filename)
    print("Printing Predicions")
    results = model.predict_classes(X, batch_size=1)
    print(results)
    print("#########################################")
    print("#########Computation Complete############")
    print("#########################################")