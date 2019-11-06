import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.python.framework import test_util
import matplotlib.pyplot as plt
import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

test_util.IsMklEnabled()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 7777  # test image from set
print(y_train[image_index])  # should be 8

# testing selected image from train set
# plt.imshow(x_train[image_index], cmap='Greys')
# plt.show()

print(x_train.shape)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# output values must be float, so we can get decimal points after division, so:
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalizing the RGB codes by dividing it to the max RGB value
x_train = x_train / 255
y_train = y_train / 255

print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Creating a sequential model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=2)

model.evaluate(x_test, y_test)

image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28), cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())
