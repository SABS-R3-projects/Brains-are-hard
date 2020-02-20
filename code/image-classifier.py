from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np




def build_model():
    '''Makes categorical classification model.
    Stolen from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

    :return:
    '''
    model = Sequential()
    model.add(Conv1D(32, 3, input_shape=(64, 30)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(32, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3)) # 3 different classes
    model.add(Activation('softmax')) # better than sigmoid for classification

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def train_model(model):
    batch_size = 16

    npz_file = np.load('training_data.npz')
    ylabel = npz_file['labels']
    xdata = npz_file['data']

    model.fit(
        xdata,
        ylabel,
        batch_size=batch_size,
        epochs=10)

    npz_file = np.load('testing_data.npz')
    test_y = npz_file['labels']
    test_x = npz_file['data']

    pred_y = model.predict(test_x)

    print(pred_y)


#    model.save_weights('weights/first_try.h5')  # always save your weights

if __name__ == "__main__":
    print(os.getcwd())
    model = build_model()
    train_model(model)