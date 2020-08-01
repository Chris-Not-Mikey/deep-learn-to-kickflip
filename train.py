import csv
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers import LSTM
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D
from keras.layers import TimeDistributed, GRU, Dense, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.image import load_img





def read_video(video):
    frame_count = []
    frames = []
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    while success:     
        success,image = vidcap.read()
        if success == False:
            continue

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(img)
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



# CNN training
def train_model(X, y, cnn_filename):

    # Create, train, compile and save model
    model = Sequential()
    model.add(Reshape(target_shape = (1080, 1920, 3, 1), input_shape = (1080, 1920, 3)))
    model.add(Dense(units=64, activation='relu'))
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.fit(X, y, epochs=1, batch_size=1, validation_data=y)

    # save model
    model.save(cnn_filename)

    # self.results = model.predict_classes(X_test)
    # self.write_results_to_csv(start)

    # INSHAPE=(1, 1080, 1920, 3) # (5, 112, 112, 3)
    # model = action_model(INSHAPE, len(y))
    # optimizer = keras.optimizers.Adam(0.001)
    # model.compile(
    #     optimizer,
    #     'categorical_crossentropy',
    #     metrics=['acc']
    # )

    # EPOCHS=50
    # # create a "chkp" directory before to run that
    # # because ModelCheckpoint will write models inside
    # callbacks = [
    #     keras.callbacks.ReduceLROnPlateau(verbose=1),
    #     keras.callbacks.ModelCheckpoint(
    #         'chkp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
    #         verbose=1),
    # ]
    # print(X.shape)
    
    # model.fit_generator(
    #     X,
    #     validation_data=y,
    #     verbose=1,
    #     epochs=EPOCHS,
       
    # )


def build_convnet(shape=(1080, 1920, 3)):
    momentum = .9
    model = keras.Sequential()
    model.add(Conv2D(64, (3,3), input_shape=shape,
        padding='same', activation='relu'))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    model.add(MaxPool2D())
    
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3,3), padding='same', activation='relu'))
    model.add(BatchNormalization(momentum=momentum))
    
    # flatten...
    model.add(GlobalMaxPool2D())
    return model



def action_model(shape=(1, 1080, 1920, 3), nbout=3):
    # Create our convnet with (112, 112, 3) input shape
    convnet = build_convnet(shape[1:])
    
    # then create our final model
    model = keras.Sequential()    # add the convnet with (5, 112, 112, 3) shape
    model.add(TimeDistributed(convnet, input_shape=shape))    # here, you can also use GRU or LSTM
    model.add(GRU(64))    # and finally, we make a decision network
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))
    return model


if __name__ == "__main__":


    file = "./videos/train/training_video.mp4"
    frames, old_count = read_video(file)
    #write_to_csv_file("validate", frame_count, frames)
    frame_count, new_frames, kickflip_bools = read_csv_file("trained_kickflip_results")


    cnn_filename = "./cnn_models"
    
    X = np.array(frames)
    print(len(X))
    
    # print(X.dtype)
    y = np.array(kickflip_bools, dtype=np.int64)
    y_len = len(y) - 1
    X_mod = X[920:]
    y_mod = y[920:y_len]
    print(len(y_mod))
    train_model(X_mod, y_mod, cnn_filename)

    # print(kickflip_bools)

    print("#########################################")
    print("#########Computation Complete############")
    print("#########################################")