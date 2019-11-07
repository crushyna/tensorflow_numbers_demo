import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 7777  # test image from set
print(y_train[image_index])     # should be 8

# testing selected image from train set
plt.imshow(x_train[image_index], cmap='Greys')
plt.show()

print(x_train.shape)