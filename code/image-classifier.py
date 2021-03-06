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
    model.add(Conv2D(16, 3, input_shape=(64, 30, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=4))

    #model.add(Conv2D(16, 3))
    #model.add(Activation('relu'))
    #model.add(Conv2D(16, 3))
    #model.add(Activation('relu'))
    #model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.2)) # gets rid of this % of nodes
    model.add(Dense(3)) # 3 different classes
    model.add(Activation('softmax')) # better than sigmoid for classification

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[acc_metric],)

    return model

def train_model(model):
    batch_size = 32
    num_epochs = 200

    npz_file = np.load('training_data.npz')
    ylabel = npz_file['labels']
    xdata = npz_file['data']

    history = model.fit(
        xdata,
        ylabel,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.2
        )

    npz_file = np.load('testing_data.npz')
    test_y = npz_file['labels']
    test_x = npz_file['data']

    pred_y = model.predict(test_x)
    rounded_pred = np.argmax(pred_y, axis=1)
    rounded_test = np.argmax(test_y, axis=1) # https://stackoverflow.com/questions/54589669/

    auc = metrics.roc_auc_score(test_y, pred_y)
    acc = metrics.accuracy_score(rounded_test, rounded_pred)
    cm = metrics.confusion_matrix(rounded_test, rounded_pred)
    print('auc is ' + str(auc))
    print('acc is ' + str(acc))
    print('confusion matrix:-')
    print(cm)

    pyplot.plot(history.history['val_categorical_accuracy'], linewidth=3)
    pyplot.plot(history.history[acc_metric], linewidth=3)
    pyplot.plot(history.history['val_loss'], linewidth=3)
    pyplot.plot(history.history['loss'], linewidth=3, color='purple')
    pyplot.ylabel('acc')
    pyplot.xlabel('epoch')
    pyplot.xlim(0,num_epochs)
    pyplot.ylim(0)
    pyplot.title('Categorical Accuracy & loss for classifier')
    pyplot.legend(['validation accuracy metric', 'training accuracy metric', 'validation loss', 'training loss'], loc='lower left')
    pyplot.savefig('Cat acc loss.png')
    pyplot.show()

#    model.save_weights('weights/first_try.h5')

if __name__ == "__main__":
    acc_metric = 'categorical_accuracy'
    model = build_model()
    train_model(model)