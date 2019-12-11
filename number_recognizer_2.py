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

    (X_train, y_train), (X_test, y_test) = mnist.load_data(path='MNIST_data')

    @staticmethod
    def load_dataset():
        pass

    @staticmethod
    def train_neural_network():
        # get the MNIST dataset
        # split the dataset into training and testing datasets
        # (X_train, y_train), (X_test, y_test) = mnist.load_data(path='MNIST_data')

        # TODO: shape isn't changing after adding new image to dataset. Is merging even working?
        print(NeuralNetwork.X_train.shape)
        print(NeuralNetwork.X_test.shape)
        print(NeuralNetwork.y_train.shape)
        print(NeuralNetwork.y_test.shape)

        # flatten 28*28 images to a 784 vector for each image
        num_pixels = NeuralNetwork.X_train.shape[1] * NeuralNetwork.X_train.shape[2]
        NeuralNetwork.X_train = NeuralNetwork.X_train.reshape((NeuralNetwork.X_train.shape[0], num_pixels)).astype('float32')
        NeuralNetwork.X_test = NeuralNetwork.X_test.reshape((NeuralNetwork.X_test.shape[0], num_pixels)).astype('float32')

        # normalize inputs from 0-255 to 0-1
        NeuralNetwork.X_train = NeuralNetwork.X_train / 255
        NeuralNetwork.X_test = NeuralNetwork.X_test / 255

        # each matrix in the training/test set needs to be “unrolled”
        NeuralNetwork.X_train = NeuralNetwork.X_train.reshape(-1, 784)
        NeuralNetwork.X_test = NeuralNetwork.X_test.reshape(-1, 784)

        # convert train and test outputs to one hot encoding
        NeuralNetwork.y_train = to_categorical(NeuralNetwork.y_train, 10)
        NeuralNetwork.y_test = to_categorical(NeuralNetwork.y_test, 10)

        model = Sequential()
        model.add(Dense(784, input_dim=28 * 28, activation='relu'))
        # model.add(Dropout(0.1, seed=3))
        model.add(Flatten())
        model.add(Dense(1568, activation='relu'))
        # model.add(Dropout(0.1, seed=3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='softmax'))

        # model.compile configures the learning process
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # epochs is the number of passes through the training data
        # batch size is the number of samples that are sent through the network
        model.fit(NeuralNetwork.X_train, NeuralNetwork.y_train, epochs=1, shuffle=True, verbose=2, batch_size=2000)

        # run neural network on test data
        test_error_rate = model.evaluate(NeuralNetwork.X_test, NeuralNetwork.y_test, verbose=2)
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
