from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn import metrics
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from matplotlib import pyplot



def build_model():
    '''Makes categorical classification model.
    Stolen from https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

    :return:
    '''
    model = Sequential()
    model.add(Conv2D(32, 3, input_shape=(64, 30, 4)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(32, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3)) # 3 different classes
    model.add(Activation('softmax')) # better than sigmoid for classification

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=[acc_metric],)

    return model

def train_model(model):
    batch_size = 16

    npz_file = np.load('training_data.npz')
    ylabel = npz_file['labels']
    xdata = npz_file['data']

    history = model.fit(
        xdata,
        ylabel,
        batch_size=batch_size,
        epochs=10)

    npz_file = np.load('testing_data.npz')
    test_y = npz_file['labels']
    test_x = npz_file['data']

    pred_y = model.predict(test_x)

    auc = metrics.roc_auc_score(test_y, pred_y)
    print('auc is ' + str(auc))

    pyplot.plot(history.history[acc_metric])
    #pyplot.plot(history.history['mse'])
    pyplot.ylabel('auc')
    pyplot.xlabel('epoch')
    pyplot.show()

#    model.save_weights('weights/first_try.h5')

if __name__ == "__main__":
    acc_metric = 'categorical_accuracy'
    model = build_model()
    train_model(model)