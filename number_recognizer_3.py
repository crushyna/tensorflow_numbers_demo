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

    seed = 7
    np.random.seed(seed)

    def __init__(self):
        self.dataset = NeuralNetwork.load_dataset()
        # 0:(0:X_train, 1:y_train), 1:(0:X_test, 1:y_test) = mnist.load_data(path='MNIST_data')
        self.X_train = self.dataset[0][0]
        self.y_train = self.dataset[0][1]
        self.X_test = self.dataset[1][0]
        self.y_test = self.dataset[1][1]

    def reshape(self):
        self.X_train = self.X_train.reshape(self.X_train.shape[0],
                                            self.X_train.shape[1],
                                            self.X_train.shape[2], 1).astype('float32')
        self.X_test = self.X_test.reshape(self.X_test.shape[0],
                                          self.X_test.shape[1],
                                          self.X_test.shape[2], 1).astype('float32')

        return self.X_train, self.X_test

    @staticmethod
    def load_dataset():
        (X_train, y_train), (X_test, y_test) = mnist.load_data(path='MNIST_data')

        return (X_train, y_train), (X_test, y_test)

    def train_neural_network(self):

        self.reshape()

        print(self.X_train.shape)
        print(self.X_test.shape)
        print(self.y_train.shape)
        print(self.y_test.shape)

        # normalize inputs from 0-255 to 0-1
        self.X_train /= 255
        self.X_test /= 255

        # one hot encode
        number_of_classes = 10
        self.y_train = np_utils.to_categorical(self.y_train, number_of_classes)
        self.y_test = np_utils.to_categorical(self.y_test, number_of_classes)

        # create model
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(self.X_train.shape[1],
                                                  self.X_train.shape[2], 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(number_of_classes, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        model.fit(self.X_train, self.y_train,
                  validation_data=(self.X_test, self.y_test), epochs=2, batch_size=400)

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
        print(features_data.shape)
        label_data = np.append(label_data, [image_label], axis=0)
        print(label_data.shape)

        return features_data, label_data

    def merge_images(self, label: int, image_file: str):

        # TODO: there's an error somewhere here! It does not update the set!

        self.load_images_to_data(label, image_file, self.X_train, self.y_train)
        self.load_images_to_data(label, image_file, self.X_test, self.y_test)

        return self.X_train, self.y_train, self.X_test, self.y_test

        '''
        print(self.X_train.shape)
        print(self.y_train.shape)
        print(self.X_test.shape)
        print(self.y_test.shape)
        '''