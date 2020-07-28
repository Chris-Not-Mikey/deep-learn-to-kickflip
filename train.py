import csv
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence





def read_video(video):
    frame_count = []
    frames = []
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    while success:     
        success,image = vidcap.read()
        frames.append(image)
        frame_count.append(count)
        count += 1


    return frames, frame_count

# Write blink prediction data to a csv file
def write_to_csv_file(name, frame_count, frames):
    path = './csv_files/' + name + '_kickflip_results.csv'
    with open(path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',',
                        quotechar='"', quoting=csv.QUOTE_MINIMAL)

        counter = 0
        csv_writer.writerow(["frame", "image", "kickflip"])
        for i in frame_count:
            row = []
            row.append(i)
            row.append(frames[counter])
            row.append(0)

            csv_writer.writerow([row[0], row[1], row[2]])
            counter = counter + 1

    csvfile.close()

def read_csv_file(name):
   
    frame_count = []
    frames = []
    kickflip_bools = []

    path = './csv_files/train/' + name + '.csv'
    with open(path) as s:
        reader = csv.reader(s)

        counter = 0
        for row in reader:
            if counter != 0:
                frame_count.append(row[0])
                frames.append(row[1])
                kickflip_bools.append(row[2])

            counter = counter + 1

    s.close()

    return frame_count, frames, kickflip_bools



if __name__ == "__main__":


    file = "./videos/validate/validation_video.mp4"
    frames, old_count = read_video(file)
    #write_to_csv_file("validate", frame_count, frames)
    frame_count, new_frames, kickflip_bools = read_csv_file("trained_kickflip_results")


    X = np.array(frames, dtype=object)
    y = np.array(kickflip_bools, dtype=np.int64)
    #train_model(cnn_filename, X_train, y_train, X_test, y_test, start)

    print(kickflip_bools)

    print("#########################################")
    print("#########Computation Complete############")
    print("#########################################")