import os
import glob2
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import cv2
import numpy as np
from matplotlib import pyplot as plt

from capsulenet import CapsNet
from config import ENCODED2LABEL, BATCH_SIZE, IMAGE_SIZE


def classify_single_image(img):

    img = cv2.resize(src=img, dsize=IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    imgs = np.array([img for i in range(BATCH_SIZE)])
    
    _, model, _ = CapsNet(input_shape=imgs.shape[1:],
                    n_class=len(ENCODED2LABEL),
                    routings=args['routings'],
                    batch_size=BATCH_SIZE)

    print("[INFO] Loading model weights from {}".format(args['weights']))
    model.load_weights(args['weights'])
    
    print("[INFO] Inferencing on image ...")
    y_pred, x_recon = model.predict(imgs)
    
    class_id = np.argmax(y_pred[0])
    score = y_pred[0][class_id]

    return class_id, score


def classify_image_folder(folder_path):

    
    image_paths = glob2.glob(os.path.join(folder_path, '*.png'))
    imgs = [cv2.resize(src=cv2.imread(image_path), dsize=IMAGE_SIZE, interpolation=cv2.INTER_AREA) for image_path in image_paths]

    n_images = len(imgs)
    n_remain_imgs = BATCH_SIZE - (n_images % BATCH_SIZE)
    [imgs.append(imgs[0]) for _ in range(n_remain_imgs)]
    imgs = np.array(imgs)

    _, model, _ = CapsNet(input_shape=imgs.shape[1:],
                    n_class=len(ENCODED2LABEL),
                    routings=args['routings'],
                    batch_size=BATCH_SIZE)

    print("[INFO] Loading model weights from {}".format(args['weights']))
    model.load_weights(args['weights'])
    
    print("[INFO] Inference ...")
    y_pred, x_recon = model.predict(imgs)

    class_ids = np.argmax(y_pred[:len(y_pred)-n_remain_imgs, :], axis=1)
    scores = np.max(y_pred[:len(y_pred)-n_remain_imgs, :], axis=1)
    
    return class_ids, scores


def main(args):

    # Load dataset
    if not os.path.exists(args['image_path']):
        print("[ERROR] The path of image is INVALID !")
        exit(0)

    print("[INFO] Loading image from {} ...".format(args['image_path']))

    if os.path.isfile(args['image_path']):
        img = cv2.imread(args['image_path'])
        label, score = classify_single_image(img=img)
        print("--> The class of image is {} with score {}".format(ENCODED2LABEL[label], score))
    elif os.path.isdir(args['image_path']):
        classify_image_folder(folder_path=args['image_path'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training Capsule classifier on custom dataset.")
    

    parser.add_argument('--image_path', default='./test_images/172_1.png', type=str,
                        help='The path of the predicted image.')
    parser.add_argument('-w', '--weights', default='weights/',
                        help="The path of model weights for testing")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    args = vars(parser.parse_args())
    
    # Runing with input arguments
    main(args)

    print("[DONE] Inference progress stop !!")