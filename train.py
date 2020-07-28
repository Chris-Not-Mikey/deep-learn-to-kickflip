import csv
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
        frame_count.append(frame_count)
        count += 1


    return frames, frame_count


    # Write blink prediction data to a csv file
    def write_to_csv_file(self, name, test_start):
        path = '../../data/blink_outputs/' + name + '_blink_results.csv'
        with open(path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)

            counter = test_start
            csv_writer.writerow(["time", "EAR", "Blink"])
            for i in self.results:
                row = []
                row.append(self.ear_independent_time[counter])
                row.append(self.ear_avg_list[counter])
                row.append(i[0][0])

                csv_writer.writerow([row[0], row[1], row[2]])
                counter = counter + 1

        csvfile.close()


if __name__ == "__main__":


    file = "./videos/train/training_video.mp4"
    frames, frame_count = read_video(file)




    print("#########################################")
    print("#########Computation Complete############")
    print("#########################################")