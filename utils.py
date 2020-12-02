import numpy as np
from matplotlib import pyplot as plt
import csv
import math
import pandas
import glob2
import cv2
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import IMAGE_SIZE, BATCH_SIZE, ROUTINGS


def plot_log(filename, show=True):

    data = pandas.read_csv(filename)

    fig = plt.figure(figsize=(4,6))
    
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for key in data.keys():
        if key.find('loss') >= 0 and not key.find('val') >= 0:  # training loss
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training loss')

    fig.add_subplot(212)
    for key in data.keys():
        if key.find('acc') >= 0:  # acc
            plt.plot(data['epoch'].values, data[key].values, label=key)
    plt.legend()
    plt.title('Training and validation accuracy')

    fig.savefig('result/log.png')
    if show:
        plt.show()


def combine_images(generated_images, height=None, width=None):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
    elif width is not None and height is None:  # height not given
        height = int(math.ceil(float(num)/width))
    elif height is not None and width is None:  # width not given
        width = int(math.ceil(float(num)/height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def load_dataset(path):
    
    X = []
    y = []

    image_paths = sorted(glob2.glob(os.path.join(path, 'images', '*.png')))
    for image_path in image_paths:
        img = cv2.resize(src=cv2.imread(image_path), dsize=IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        X.append(img)
    with open(os.path.join(path, 'label.txt'), 'r') as f:
        data_labels = f.readlines()
        for line in data_labels:
            y.append(line.rstrip().split()[1])
    return np.array(X), np.array(y)


def split_dataset(data, label, ratio):

    from sklearn.model_selection import train_test_split
    
    test_size = int(len(data)*ratio)
    while test_size % 512 != 0:
        test_size += 1

    (x_train, x_test, y_train, y_test) = train_test_split(data, label, test_size=test_size, random_state=18521489)
    x_train = x_train/255.
    x_test = x_test/255.
    y_train = to_categorical(np.reshape(y_train, (-1, 1)).astype('float32'))
    y_test = to_categorical(np.reshape(y_test, (-1, 1)).astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)


def data_generator(x, y, batch_size, shift_fraction=0.):
    train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                        height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
    generator = train_datagen.flow(x, y, batch_size=batch_size, shuffle=True)
    x_batch_, y_batch_ = generator.next()
    while 1:
        x_batch, y_batch = generator.next()
        if x_batch.shape[0] < batch_size:
            yield (x_batch_, y_batch_), (y_batch_, x_batch_)
        else:
            yield (x_batch, y_batch), (y_batch, x_batch)


def print_info(args):
    
    print("The input argument information for training Capsule network classifier.".upper().center(100))
    print(str("-"*63).center(100))
    print("|{:<30}|{:<30}|".format("Data path", args['data_path']).center(100))
    print("|{:<30}|{:<30}|".format("Image size", str(IMAGE_SIZE)).center(100))
    print("|{:<30}|{:<30}|".format("Epochs", args['epochs']).center(100))
    print("|{:<30}|{:<30}|".format("Batch size", BATCH_SIZE).center(100))
    print("|{:<30}|{:<30}|".format("Test size", args['ratio']).center(100))
    print("|{:<30}|{:<30}|".format("Learning rate", args['lr']).center(100))
    print("|{:<30}|{:<30}|".format("Learning rate decay", args['lr_decay']).center(100))
    print("|{:<30}|{:<30}|".format("lam_recons", args['lam_recons']).center(100))
    print("|{:<30}|{:<30}|".format("Number of routings", ROUTINGS).center(100))
    print("|{:<30}|{:<30}|".format("Shift fraction", args['shift_fraction']).center(100))
    print("|{:<30}|{:<30}|".format("Saving directory", args['save_dir']).center(100))
    print("|{:<30}|{:<30}|".format("Model weights directory", str(args['weights'])).center(100))
    print(str("-"*63).center(100))


if __name__=="__main__":
    plot_log('result/log.csv')



