from os import listdir

import PIL.Image as Image
import time
import numpy as np
import tensorflow as tf
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
        self.dataset = object
        # 0:(0:X_train, 1:y_train), 1:(0:X_test, 1:y_test) = mnist.load_data(path='MNIST_data')
        self.X_train = np.ndarray
        self.y_train = np.ndarray
        self.X_test = np.ndarray
        self.y_test = np.ndarray

    def reshape(self):
        self.X_train = self.X_train.reshape(self.X_train.shape[0],
                                            self.X_train.shape[1],
                                            self.X_train.shape[2], 1).astype('float32')
        self.X_test = self.X_test.reshape(self.X_test.shape[0],
                                          self.X_test.shape[1],
                                          self.X_test.shape[2], 1).astype('float32')

        return self.X_train, self.X_test

    def load_clean_dataset(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data(path='MNIST_data')

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        print("Clean dataset loaded!")

        return self.X_train, self.y_train, self.X_test, self.y_test

    def save_working_dataset(self):
        self.X_train.dump("working_dataset/X_train.npy")
        self.y_train.dump("working_dataset/y_train.npy")
        self.X_test.dump("working_dataset/X_test.npy")
        self.y_test.dump("working_dataset/y_test.npy")
        print("Working dataset saved!")

    def load_working_dataset(self):
        self.X_train = np.load("working_dataset/X_train.npy", allow_pickle=True)
        self.y_train = np.load("working_dataset/y_train.npy", allow_pickle=True)
        self.X_test = np.load("working_dataset/X_test.npy", allow_pickle=True)
        self.y_test = np.load("working_dataset/y_test.npy", allow_pickle=True)
        print("Working dataset loaded!")

        return self.X_train, self.y_train, self.X_test, self.y_test

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
        NEW_y_train = np_utils.to_categorical(self.y_train, number_of_classes)
        NEW_y_test = np_utils.to_categorical(self.y_test, number_of_classes)

        # create model
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(self.X_train.shape[1],
                                                  self.X_train.shape[2], 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(number_of_classes, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        model.fit(self.X_train, NEW_y_train,
                  validation_data=(self.X_test, NEW_y_test), epochs=1, batch_size=300)

        # Save the model
        print("Saving model...")
        model.save('trainedMNISTModel.h5')
        print("Saved!")
        self.save_working_dataset()
        time.sleep(1)

        return 1

    @staticmethod
    def predict_number(image_of_number: str):
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

    def load_images_to_data(self, image_label: str, image_file: str):
        img = Image.open(image_file).convert("L")
        img = img.resize((28, 28))
        im2arr = np.array(img)
        im2arr = im2arr / 255
        im2arr = im2arr.reshape(1, 28, 28, 1)

        self.X_train = np.append(self.X_train, im2arr, axis=0)
        self.X_test = np.append(self.X_test, im2arr, axis=0)
        print(self.X_train.shape)
        print(self.X_test.shape)

        self.y_train = np.append(self.y_train, [image_label], axis=0)
        self.y_test = np.append(self.y_test, [image_label], axis=0)
        print(self.y_train.shape)
        print(self.y_test.shape)

    def merge_images(self, label: str, image_file: str):
        self.reshape()
        self.load_images_to_data(label, image_file)

        return self.X_train, self.y_train, self.X_test, self.y_test


# visualize one number with pixel values
def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y], 2)), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y] < thresh else 'black')
    # make it happen!
