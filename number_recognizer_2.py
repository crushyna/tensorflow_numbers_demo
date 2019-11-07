import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import os
import PIL.Image as Image
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.utils.np_utils import to_categorical
from keras.models import load_model


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

    # plot the first 4 images
    '''
    for i in range(1, 5):
        plt.imshow(x_train[i])
        plt.show()
    '''

    '''
    img3 = imread('image.png')
    print(img3.shape)

    img4 = np.asarray(Image.open("image.png").convert('L')).ravel()
    img4 = img4.reshape(-1, 784)
    print(img4.shape)

    img5 = np.invert(Image.open("image.png").convert('L')).ravel()
    img5 = img5.reshape(-1, 784)
    print(img5.shape)
    '''

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


'''
prediction = model.predict(img4)
plt.imshow(img4)
plt.ylabel("Predicted Value: " + str(np.argmax(prediction)))
plt.xlabel("Actual Value: " + str(y_validate[i]))
plt.show()
'''

'''
img1 = Image.open('image.png').convert('L')
img1.save("temp.png")
img1.show()
img2 = np.asarray(Image.open('image.png').convert('L'))
'''


# img3 = imread('image.png')

def predict_number(image_of_number):

    img3 = imread(image_of_number)
    print(img3.shape)

    img4 = np.asarray(Image.open(image_of_number).convert('L')).ravel()
    img4 = img4.reshape(-1, 784)
    print(img4.shape)

    img5 = np.invert(Image.open(image_of_number).convert('L')).ravel()
    img5 = img5.reshape(-1, 784)
    print(img5.shape)

    # load trained model
    model = load_model("trainedMNISTModel.h5")

    # actual prediction
    prediction1 = model.predict(img4)
    plt.imshow(img4)
    print("Prediction for test image 1:", np.squeeze(prediction1))

    prediction2 = model.predict(img5)
    plt.imshow(img5)
    print("Prediction for test image 2:", np.squeeze(prediction2))

    return prediction2


