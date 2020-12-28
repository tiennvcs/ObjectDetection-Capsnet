import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import argparse
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import IMAGE_SIZE, BATCH_SIZE, SAVE_FREQ
from utils import print_info, load_dataset, split_dataset, data_generator, plot_log

K.set_image_data_format('channels_last')


def build_model(input_shape, n_class):

    model = Sequential()
    model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), 
                                    strides=(1, 1), padding='same', 
                                    activation='relu', input_shape=input_shape))
    #model.add(layers.MaxPooling2D(pool_size=(1, 1)))
    model.add(layers.Conv2D(filters=256, kernel_size=(5, 5), 
                                    strides=(1, 1), padding='same', 
                                    activation='relu'))
    #model.add(layers.MaxPooling2D(pool_size=(1, 1)))
    model.add(layers.Conv2D(filters=128, kernel_size=(5, 5), 
                                    strides=(1, 1), padding='same', 
                                    activation='relu'))
    #model.add(layers.MaxPooling2D(pool_size=(1, )))

    model.add(layers.Flatten())
    model.add(layers.Dense(328, activation='relu'))
    model.add(layers.Dense(192, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(n_class, activation='softmax'))
    
    return model


def train(model, data, args):
    
    (x_train, y_train), (x_test, y_test) = data        

    my_callbacks = [
        callbacks.EarlyStopping(patience=2),
        callbacks.CSVLogger(os.path.join(args['save_dir'], 'log.csv')),
        callbacks.ModelCheckpoint(filepath=os.path.join(args['save_dir'], 'model.{epoch:02d}-{val_loss:.2f}.h5')),
        callbacks.LearningRateScheduler(schedule=lambda epoch: args['lr']*(args['lr_decay'] ** epoch)),
    ]

    # log = callbacks.CSVLogger(os.path.join(args['save_dir'], 'log.csv'))    
    # saving_path = os.path.join(args['save_dir'], 'weights-{epoch:02d}.h5')

    # checkpoint = callbacks.ModelCheckpoint(saving_path, monitor='val_loss', 
    #                                         save_freq=SAVE_FREQ*int(y_train.shape[0]/BATCH_SIZE), save_best_only=False, 
    #                                         save_weights_only=True, verbose=1)
    # lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args['lr']*(args['lr_decay'] ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args['lr']),
                  loss=['categorical_crossentropy'],
                  metrics=['accuracy'])
                  
    # Training with data augmentation. If shift_fraction=0., no augmentation.
    # model.fit(data_generator(x_train, y_train, BATCH_SIZE, args['shift_fraction']),
    #         epochs=args['epochs'],
    #         steps_per_epoch=int(y_train.shape[0]/BATCH_SIZE),
    #         validation_data=data_generator(x_test, y_test, BATCH_SIZE, args['shift_fraction']), batch_size=BATCH_SIZE,
    #         validation_freq=1,
    #         #callbacks=[log, checkpoint, lr_decay]
    #         callbacks=my_callbacks)
    datagen = ImageDataGenerator(
                width_shift_range=args['shift_fraction'],
                height_shift_range=args['shift_fraction'],
                )
    generator = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

    model.fit(generator,  batch_size=BATCH_SIZE,
                steps_per_epoch=int(y_train.shape[0]/BATCH_SIZE),
                validation_data=((x_test, y_test), (y_test, x_test)),
                validation_freq=1, 
                callbacks=my_callbacks)

    model.save_weights(args['save_dir'] + '/final_trained_model.h5')
    print('Trained model saved to \'%s/final_trained_model.h5\'' % args['save_dir'])

    plot_log(args['save_dir'] + '/log.csv', show=False)


def main(args):

    if not os.path.exists(args['save_dir']):
        os.mkdir(args['save_dir'])

    # Set the fixed random seed
    np.random.seed(18521489)

    if not os.path.exists(args['data_path']):
        print("[ERROR] Invalid data directory")
        exit(0)

    X, y = load_dataset(args['data_path'])
    (x_train, y_train), (x_test, y_test) = split_dataset(data=X, label=y, ratio=args['ratio'])

    # print data information
    print(str("{:<40}|{:<30}".format("The number of training examples", len(x_train))).center(100))
    print(str("{:<40}|{:<30}".format("The number of valid examples", len(x_test))).center(100))
    print(str("{:<40}|{:<30}".format("The number of classes", len(np.unique(np.argmax(y_train, 1))))).center(100))
    #time.sleep(5)

    model = build_model(input_shape=x_train.shape[1:],
                    n_class=len(np.unique(np.argmax(y_train, 1))))

    model.summary()

    # Load model weights from the path                    
    if args['weights'] is not None:
        model.load_weights(args['weights'])
    
    train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Capsule classifier on custom dataset.")

    parser.add_argument('--data_path', default=None, type=str,
                        help='The path of training image folder')
    parser.add_argument('--ratio', default=0.2, type=float,
                        help='The ratio splitting data into validation set and training set.')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recons', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('--shift_fraction', default=0.15, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='result/')
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    # End default optional arguments

    args = vars(parser.parse_args())
    args['save_dir'] = 'CNN_' + args['save_dir']
    print("The input argument information for training Convolutional neural networks.".upper().center(100))
    print(str("-"*63).center(100))
    print("|{:<30}|{:<30}|".format("Data path", args['data_path']).center(100))
    print("|{:<30}|{:<30}|".format("Image size", str(IMAGE_SIZE)).center(100))
    print("|{:<30}|{:<30}|".format("Epochs", args['epochs']).center(100))
    print("|{:<30}|{:<30}|".format("Batch size", BATCH_SIZE).center(100))
    print("|{:<30}|{:<30}|".format("Test size", args['ratio']).center(100))
    print("|{:<30}|{:<30}|".format("Learning rate", args['lr']).center(100))
    print("|{:<30}|{:<30}|".format("Learning rate decay", args['lr_decay']).center(100))
    print("|{:<30}|{:<30}|".format("lam_recons", args['lam_recons']).center(100))
    print("|{:<30}|{:<30}|".format("Shift fraction", args['shift_fraction']).center(100))
    print("|{:<30}|{:<30}|".format("Saving directory", args['save_dir']).center(100))
    print("|{:<30}|{:<30}|".format("Model weights directory", str(args['weights'])).center(100))
    print(str("-"*63).center(100))

    main(args)