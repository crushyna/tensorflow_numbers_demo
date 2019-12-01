import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Flatten
from keras.utils.np_utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical


class NeuralNetwork:

    @staticmethod
    def train_neural_network():
        # get the MNIST dataset
        # split the dataset into training and testing datasets
        (X_train, y_train), (X_test, y_test) = mnist.load_data(path='MNIST_data')

        # flatten 28*28 images to a 784 vector for each image
        num_pixels = X_train.shape[1] * X_train.shape[2]
        X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
        X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

        # normalize inputs from 0-255 to 0-1
        X_train = X_train / 255
        X_test = X_test / 255

        # each matrix in the training/test set needs to be “unrolled”
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)

        # convert train and test outputs to one hot encoding
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        model = Sequential()
        model.add(Flatten(784, input_dim=28 * 28, activation='relu'))
        model.add(Dropout(0.1, seed=3))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.1, seed=3))
        # model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.1, seed=3))
        model.add(Dense(10, activation='softmax'))

        # model.compile configures the learning process
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # epochs is the number of passes through the training data
        # batch size is the number of samples that are sent through the network
        model.fit(X_train, y_train, epochs=3, shuffle=True, verbose=2, batch_size=128)

        # run neural network on test data
        test_error_rate = model.evaluate(X_test, y_test, verbose=2)
        print(model.metrics_names)
        print(test_error_rate)

        model.save("trainedMNISTModel.h5")

        return model

    @staticmethod
    def predict_number(image_of_number):
        img4 = np.asarray(Image.open(image_of_number).convert('L')).ravel()
        img4 = img4.reshape(-1, 784)
        print(img4.shape)

        img5 = np.invert(Image.open(image_of_number).convert('L')).ravel()
        img5 = img5.reshape(-1, 784)
        print(img5.shape)

        # load trained model
        model = load_model('trainedMNISTModel.h5')

        # actual prediction
        prediction1 = model.predict(img4)
        result_index1 = np.where((np.squeeze(prediction1)) == 1)
        plt.imshow(img4)
        print("Prediction for asarray image:", result_index1[0])

        prediction2 = model.predict(img5)
        plt.imshow(img5)
        result_index2 = np.where((np.squeeze(prediction2)) == 1)
        print("Prediction for invert image:", result_index2[0])

        return result_index2[0]

    @staticmethod
    def load_images_to_data(image_label: int, image_file: str, features_data, label_data):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        img = Image.open(image_file).convert("L")
        img = np.resize(img, (28, 28, 1))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1, 28, 28, 1)
        features_data = np.append(features_data, im2arr, axis=0)
        # TODO: resolve error: the array at index 0 has 3 dimension(s) and the array at index 1 has 4 dimension(s)
        label_data = np.append(label_data, [image_label], axis=0)
        return features_data, label_data

    @staticmethod
    def merge_images(label: int, image_file: str):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        NeuralNetwork.load_images_to_data(label, image_file, X_train, y_train)
        NeuralNetwork.load_images_to_data(label, image_file, X_test, y_test)
