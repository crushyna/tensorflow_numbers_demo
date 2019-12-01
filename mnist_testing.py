import random
from tensorflow.keras.datasets import mnist
import gzip
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import shutil

MNIST_folder = 'MNIST_data'
MNIST_raw = 'MNIST_raw'

source_train_images = f'{MNIST_folder}/train-images-idx3-ubyte.gz'
source_train_labels = f'{MNIST_folder}/train-labels-idx1-ubyte.gz'
dest_train_images = f'{MNIST_raw}/train-images-idx3-ubyte'
dest_train_labels = f'{MNIST_raw}/train-labels-idx1-ubyte'


def gunzip_shutil(source_filepath, dest_filepath, block_size=65536):
    with gzip.open(source_filepath, 'rb') as s_file, \
            open(dest_filepath, 'wb') as d_file:
        shutil.copyfileobj(s_file, d_file, block_size)


gunzip_shutil(source_train_images, dest_train_images)
gunzip_shutil(source_train_labels, dest_train_labels)

mndata = MNIST(MNIST_raw)
train_images, train_labels = mndata.load_training()

index = random.randrange(2, len(train_images))  # choose an index ;-)
print(mndata.display(train_images[index]))
