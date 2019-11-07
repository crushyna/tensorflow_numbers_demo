import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
import PIL.Image as Image
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.utils.np_utils import to_categorical


# from keras.models import load_model


def train_neural_network():
    # get the MNIST dataset
    # split the dataset into training and testing datasets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # flatten 28*28 images to a 784 vector for each image
    num_pixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape((x_train.shape[0], num_pixels)).astype('float32')
    x_test = x_test.reshape((x_test.shape[0], num_pixels)).astype('float32')

    # normalize inputs from 0-255 to 0-1
    x_train = x_train / 255
    x_test = x_test / 255

    # each matrix in the training/test set needs to be “unrolled”
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # convert train and test outputs to one hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential()
    model.add(Dense(25, input_dim=28 * 28, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # model.compile configures the learning process
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # epochs is the number of passes through the training data
    # batch size is the number of samples that are sent through the network
    model.fit(x_train, y_train, epochs=20, shuffle=True, verbose=2, batch_size=128)

    # run neural network on test data
    test_error_rate = model.evaluate(x_test, y_test, verbose=0)
    print(model.metrics_names)
    print(test_error_rate)

    model.save("trainedMNISTModel.h5")

    return model


def predict_number(image_of_number):
    img4 = np.asarray(Image.open(image_of_number).convert('L')).ravel()
    img4 = img4.reshape(-1, 784)
    print(img4.shape)

    img5 = np.invert(Image.open(image_of_number).convert('L')).ravel()
    img5 = img5.reshape(-1, 784)
    print(img5.shape)

    # load trained model
    model = load_model('trainedMNISTModel.h5')
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(x_train, y_train, epochs=1, shuffle=True, verbose=2, batch_size=128)

    # actual prediction
    prediction1 = model.predict(img4)
    result_index1 = np.where((np.squeeze(prediction1)) == 1)
    plt.imshow(img4)
    print("Prediction for test image 2:", result_index1[0])

    prediction2 = model.predict(img5)
    plt.imshow(img5)
    result_index2 = np.where((np.squeeze(prediction2)) == 1)
    print("Prediction for test image 2:", result_index2[0])

    return result_index2[0]
