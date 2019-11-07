from tkinter import Image
# import keras
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

# get the MNIST dataset
# split the dataset into training and testing datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# plot the first 4 images
'''
for i in range(1, 5):
    plt.imshow(x_train[i])
    plt.show()
'''

img3 = imread('image.png')
img4 = np.asarray(Image.open("image.png").convert('L')).ravel()
img4 = img4.reshape(-1, 784)
print(img3.shape)
print(img4.shape)

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

prediction = model.predict(img4)
plt.imshow(img4)
print("Prediction for test image:", np.squeeze(prediction))