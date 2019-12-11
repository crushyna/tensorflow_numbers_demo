import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical


class NeuralNetwork:

    def __init__(self):
        self.dataset = NeuralNetwork.load_dataset()
        # (X_train, y_train), (X_test, y_test) = mnist.load_data(path='MNIST_data')

    seed = 7
    np.random.seed(seed)

    @staticmethod
    def reshape():
        NeuralNetwork.X_train = NeuralNetwork.X_train.reshape(NeuralNetwork.X_train.shape[0],
                                                              NeuralNetwork.X_train.shape[1],
                                                              NeuralNetwork.X_train.shape[2], 1).astype('float32')
        NeuralNetwork.X_test = NeuralNetwork.X_test.reshape(NeuralNetwork.X_test.shape[0],
                                                            NeuralNetwork.X_test.shape[1],
                                                            NeuralNetwork.X_test.shape[2], 1).astype('float32')

        return NeuralNetwork.X_train, NeuralNetwork.X_test

    @staticmethod
    def load_dataset():
        (X_train, y_train), (X_test, y_test) = mnist.load_data(path='MNIST_data')

        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def train_neural_network():
        # (X_train, y_train), (X_test, y_test) = mnist.load_data(path='MNIST_data')

        # TODO: shape isn't changing after adding new image to dataset. Is merging even working?

        NeuralNetwork.reshape()

        print(NeuralNetwork.dataset.X_train.shape)
        print(NeuralNetwork.X_test.shape)
        print(NeuralNetwork.y_train.shape)
        print(NeuralNetwork.y_test.shape)

        # normalize inputs from 0-255 to 0-1
        NeuralNetwork.X_train /= 255
        NeuralNetwork.X_test /= 255

        # one hot encode
        number_of_classes = 10
        NeuralNetwork.y_train = np_utils.to_categorical(NeuralNetwork.y_train, number_of_classes)
        NeuralNetwork.y_test = np_utils.to_categorical(NeuralNetwork.y_test, number_of_classes)

        # create model
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(NeuralNetwork.X_train.shape[1],
                                                  NeuralNetwork.X_train.shape[2], 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(number_of_classes, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        model.fit(NeuralNetwork.X_train, NeuralNetwork.y_train,
                  validation_data=(NeuralNetwork.X_test, NeuralNetwork.y_test), epochs=1, batch_size=400)

        # Save the model
        model.save('trainedMNISTModel.h5')

    @staticmethod
    def predict_number(image_of_number):

        img5 = Image.open(image_of_number).convert("L")
        im2arr = np.array(img5)
        im2arr = im2arr / 255
        im2arr = im2arr.reshape((1, 28, 28, 1))
        print(im2arr.shape)

        # load trained model
        model = load_model('trainedMNISTModel.h5')

        # actual prediction
        y_pred = model.predict_classes(im2arr)
        print(y_pred)

        return y_pred

    @staticmethod
    def load_images_to_data(image_label: int, image_file: str, features_data, label_data):
        img = Image.open(image_file).convert("L")
        img = img.resize((28, 28))
        im2arr = np.array(img)
        im2arr = im2arr / 255
        im2arr = im2arr.reshape((1, 28, 28))
        features_data = np.append(features_data, im2arr, axis=0)
        label_data = np.append(label_data, [image_label], axis=0)

        return features_data, label_data

    @staticmethod
    def merge_images(label: int, image_file: str):
        # (X_train, y_train), (X_test, y_test) = mnist.load_data(path='MNIST_data')
        NeuralNetwork.load_images_to_data(label, image_file, NeuralNetwork.X_train, NeuralNetwork.y_train)
        NeuralNetwork.load_images_to_data(label, image_file, NeuralNetwork.X_test, NeuralNetwork.y_test)

        print(NeuralNetwork.X_train.shape)
        print(NeuralNetwork.y_train.shape)
        print(NeuralNetwork.X_test.shape)
        print(NeuralNetwork.y_test.shape)
