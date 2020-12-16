import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from PIL import Image


from capsulenet import CapsNet, margin_loss
from utils import print_info, load_dataset, split_dataset, \
                    data_generator, plot_log, combine_images
from config import BATCH_SIZE, ROUTINGS


K.set_image_data_format('channels_last')


def test(model, data, args):
    
    x_test, y_test = data

    y_pred, x_recon = model.predict(x_test, batch_size=BATCH_SIZE)

    test_acc = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
    img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255.
    Image.fromarray(image.astype(np.uint8)).save(os.path.join(args['output_path'], "real_and_recon.png"))
    print('--> Test acc: {}'.format(test_acc))
    print('--> Reconstructed images are saved to %s/real_and_recon.png' % args['output_path'])


def main(args):
    if not os.path.exists(args['output_path']):
        os.mkdir(args['output_path'])

    # Set the fixed random seed
    np.random.seed(18521489)

    # Load dataset
    print("[INFO] Loading data ...")
    # X, y = load_dataset(args['test_data'])
    # (x_train, y_train), (x_test, y_test) = split_dataset(data=X, label=y, ratio=0.9999999)
    x_test, y_test = load_dataset(args['test_data'], istrain=False)

    print(str("{:<40}|{:<30}".format("The number of testing examples", len(y_test))).center(100))
    print(str("{:<40}|{:<30}".format("The number of classes", len(np.unique(np.argmax(y_test, 1))))).center(100))

    _, eval_model, manipulate_model = CapsNet(input_shape=x_test.shape[1:],
                    n_class=len(np.unique(np.argmax(y_test, 1))),
                    routings=ROUTINGS,
                    batch_size=BATCH_SIZE)

    # Load model weights from the path                    
    if args['weights'] is not None:
        eval_model.load_weights(args['weights'])
        manipulate_model.load_weights(args['weights'])
    
    # eval_model.summary()
    # manipulate_model.summary()
    
    # print("[INFO] Evaluating on noise images ...")
    # manipulate_latent(model=manipulate_model, data=(x_test, y_test), 
    #               n_class=len(np.unique(np.argmax(y_test, 1))), args=args)
    print("[INFO] Evaluating on test image ...")
    test(model=eval_model, data=(x_test, y_test), args=args)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training Capsule classifier on custom dataset.")
    parser.add_argument('--test_data', default='../data', type=str,
                        help='The path of training image folder')
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of model weights for testing")
    parser.add_argument('--output_path', default='./output_test',
                        help='The directory store the output of evaluations')
    parser.add_argument('--sign', default=0, type=int, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="Traffic sign to manipulate")

    args = vars(parser.parse_args())
    
    # Runing with input arguments
    main(args)

    print("[DONE] Testing progress stop!!")