import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import argparse

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from PIL import Image


from capsulenet import CapsNet, margin_loss
from utils import print_info, load_dataset, split_dataset, \
                    data_generator, plot_log, combine_images


K.set_image_data_format('channels_last')


def test(model, data, args):
    
    x_test, y_test = data

    y_pred, x_recon = model.predict(x_test, batch_size=args['batch_size'])

    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

    img = combine_images(np.concatenate([x_test[:50], x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args['output_path'] + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args['output_path'])
    print('-'*30 + 'End: test'+ '-'*30)
    plt.imshow(plt.imread(args['output_path'] + "/real_and_recon.png"))
    plt.show()


def manipulate_latent(model, data, n_class, args):

    print('-'*30 + 'Begin: manipulate' + '-'*30)
    
    x_test, y_test = data

    index = np.argmax(y_test, 1) == args['sign']

    number = np.random.randint(low=0, high=sum(index) - 1)

    selected_indices = np.random.choice(len(y_test[index]), args['batch_size'], replace=False)

    x, y = x_test[index][selected_indices], y_test[index][selected_indices]

    noise = np.zeros([args['batch_size'], n_class, 16])
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
    Image.fromarray(image.astype(np.uint8)).save(args['output_path'] + '/manipulate-%d.png' % args['sign'])
    print('manipulated result saved to %s/manipulate-%d.png' % (args['output_path'], args['sign']))
    print('-'*30 + 'End: manipulate' + '-'*30)


def main(args):

    if not os.path.exists(args['output_path']):
        os.mkdir(args['output_path'])

    # Set the fixed random seed
    np.random.seed(18521489)

    # Load dataset
    print("[INFO] Loading data ...")
    X, y = load_dataset(args['test_data'])
    (x_train, y_train), (x_test, y_test) = split_dataset(data=X, label=y, ratio=0.5)

    _, eval_model, manipulate_model = CapsNet(input_shape=x_test.shape[1:],
                    n_class=len(np.unique(np.argmax(y_test, 1))),
                    routings=args['routings'],
                    batch_size=args['batch_size'])


    # Load model weights from the path                    
    if args['weights'] is not None:
        eval_model.load_weights(args['weights'])
        manipulate_model.load_weights(args['weights'])
    
    eval_model.summary()
    manipulate_model.summary()
    
    print("[INFO] Evaluating on noise images ...")
    manipulate_latent(model=manipulate_model, data=(x_test, y_test), 
                    n_class=len(np.unique(np.argmax(y_train, 1))), args=args)

    print("[INFO] Evaluating on test image ...")
    test(model=eval_model, data=(x_test, y_test), args=args)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training Capsule classifier on custom dataset.")
    

    parser.add_argument('--test_data', default='../data', type=str,
                        help='The path of training image folder')
    parser.add_argument('--batch_size', default=32, choices=[4, 8, 16, 32, 64, 128, 256], type=int,
                        help='The batch size for evaluations.')
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of model weights for testing")
    parser.add_argument('--output_path', default='./output_test',
                        help='The directory store the output of evaluations')
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--sign', default=0, type=int, choices=[0, 1, 2, 3, 4, 5, 6],
                        help="Traffic sign to manipulate")

    args = vars(parser.parse_args())
    
    # Runing with input arguments
    main(args)

    print("[DONE] Testing progress stop!!")