import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings, batch_size):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :param batch_size: size of batch
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape, batch_size=batch_size)
    #x = layers.Input(shape=input_shape)
    
    #print((None,*input_shape))
    #x = layers.Input(batch_input_shape = (None,*input_shape))
    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    #print(x)
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    
    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,),batch_size=batch_size)
    #y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class,batch_size=batch_size))
    #decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])
    #train_model.summary()
    #eval_model.summary()
    # manipulate model
    noise = layers.Input(shape=(n_class, 16),batch_size=batch_size)
    #noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    # return tf.reduce_mean(tf.square(y_pred))
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))

    return tf.reduce_mean(tf.reduce_sum(L, 1))

def train(model,  # type: models.Model
          data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """

    # callbacks
    log = callbacks.CSVLogger(os.path.join(args.save_dir, '/log.csv'))
    saving_path = os.path.join(args.save_dir, '/weights-{epoch:02d}.h5')
    checkpoint = callbacks.ModelCheckpoint(saving_path, monitor='val_capsnet_acc',
                                           save_best_only=False, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr*(args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})


    # Begin: Training with data augmentation ---------------------------------------------------------------------#
    def train_generator(x, y, batch_size, shift_fraction=0.):
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


    def val_generator(x, y, batch_size, shift_fraction=0.):
        val_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = val_datagen.flow(x, y, batch_size=batch_size, shuffle=True)
        x_batch_, y_batch_ = generator.next()
        while 1:
            x_batch, y_batch = generator.next()
            if x_batch.shape[0] < batch_size:
                yield (x_batch_, y_batch_), (y_batch_, x_batch_)
            else:
                yield (x_batch, y_batch), (y_batch, x_batch)

    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data    
    
    # Training with data augmentation. If shift_fraction=0., no augmentation.
    #for epochs in range(args.epochs):
    model.fit(train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
            steps_per_epoch=int(y_train.shape[0] / args.batch_size),
            epochs=args.epochs,
            validation_data=val_generator(x_test, y_test, args.batch_size, args.shift_fraction),
            validation_steps=int(y_test.shape[0]/ args.batch_size),
            validation_batch_size=args.batch_size,
            validation_freq=1,
            callbacks=[log, checkpoint, lr_decay]
            )
    #model.evaluate((x_train,y_train), (y_train,x_train), batch_size=1,
      #          #steps_per_epoch=int(y_train.shape[0] / args.batch_size),
      #          #epochs=1,
      #          #validation_data=train_generator(x_test,y_test,args.batch_size, args.shift_fraction),
      #          #validation_split=0.2,
      #          #validation_batch_size = args.batch_size,
      #          callbacks=[log, checkpoint, lr_decay]
      #          )
    # End: Training with data augmentation -----------------------------------------------------------------------#
    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv',show=True)

    return model


def test(model, data, args):
    
    x_test, y_test = data

    y_pred, x_recon = model.predict(x_test, batch_size=args.batch_size)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-'*30 + 'End: test'+ '-'*30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()


def manipulate_latent(model, data, n_class, args):

    print('-'*30 + 'Begin: manipulate' + '-'*30)
    
    x_test, y_test = data

    index = np.argmax(y_test, 1) == args.sign    
    number = np.random.randint(low=0, high=sum(index) - 1)
    selected_indices = np.random.choice(len(y_test[index]), args.batch_size, replace=False)
    x, y = x_test[index][selected_indices], y_test[index][selected_indices]

    #x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([args.batch_size, n_class, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:, :, dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.sign)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.sign))
    print('-'*30 + 'End: manipulate' + '-'*30)


if __name__ == "__main__":
    import os
    import time
    import glob2
    import numpy as np
    import cv2
    import argparse
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras import callbacks
    from utils import load_dataset, split_dataset

    parser = argparse.ArgumentParser(description="Capsule Network on custom dataset.")
    
    # My custom optional arguments
    parser.add_argument('--data_path', default='/content/data', type=str,
                        help='The path of training image folder')
    parser.add_argument('--ratio', default=0.2, choices=[0.1, 0.2], type=float,
                        help='The ratio splitting data into validation set and training set.')
    parser.add_argument('--sign', default=0, type=int, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="Traffic sign to manipulate")
    # End my custom optional arguments

    # Default optional arguments
    # setting the hyper parameters
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=32, choices=[16, 32, 64, 128, 256], type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    # End default optional arguments

    args = parser.parse_args()
    

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    # Load dataset
    X, y = load_dataset(args.data_path)
    (x_train, y_train), (x_test, y_test) = split_dataset(data=X, label=y, ratio=args.ratio)

    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings,
                                                  batch_size=args.batch_size)

    if not args.testing:
        if args.weights is not None:
            model.load_weights(args.weights)
        model.summary()
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)

    else:
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        else:
            manipulate_model.load_weights(args.weights)
            eval_model.load_weights(args.weights)

        #manipulate_latent(model=manipulate_model, data=(x_test, y_test), 
        #                n_class=len(np.unique(np.argmax(y_train, 1))), args=args)        
        test(model=eval_model, data=(x_test, y_test), args=args)
