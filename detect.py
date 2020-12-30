import os
import glob2
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

from config import ENCODED2LABEL, BATCH_SIZE, IMAGE_SIZE, ROUTINGS
from capsulenet import CapsNet

COLOR = (0, 255, 0)


def detect_single_image(image_path, detector, classifier, confidence_thresh=0.2, nms_thresh=0.4):

    if not os.path.exists(image_path):
        print("The {} is INVALID path !".format(image_path))
        exit(0)

    # Load image
    img = cv2.imread(image_path)

    # Make inference by detector
    print("[Stage 1] Localization ...")
    classes, scores, boxes = detector.detect(img, confidence_thresh, nms_thresh)

    # Crop the detect object regions    
    n_images = len(boxes)
    if n_images == 0:
        print("Don't detect any objects in image !")
        return {'class_ids':[None], 'scores':[None], 'boxes':boxes, 'detected_img':img}

    object_regions = [cv2.resize(img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]], 
                                IMAGE_SIZE, cv2.INTER_AREA) for box in boxes]
    n_remain_imgs = BATCH_SIZE - (n_images % BATCH_SIZE)
    [object_regions.append(object_regions[0]) for _ in range(n_remain_imgs)]
    object_regions = np.array(object_regions)

    # Make inference by classifier
    print("[Stage 2] Classification ...")
    y_pred, x_recon = classifier.predict(object_regions)
    class_ids = np.argmax(y_pred[:len(y_pred)-n_remain_imgs, :], axis=1)
    scores = np.max(y_pred[:len(y_pred)-n_remain_imgs, :], axis=1)

    # Draw bounding boxes and corresponding classes.
    for (classid, score, box) in zip(class_ids, scores, boxes):
        label = "%s : %.3f" % (ENCODED2LABEL[classid], score)
        img = cv2.rectangle(img, box, COLOR, 1)
        #img = cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), COLOR, 1)
        img = cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 1)
    
    cv2.imshow('Detect image', img)
    cv2.waitKey(0)
    return {'class_ids':class_ids, 'scores':scores, 'boxes':boxes, 'detected_img':img}


def detect_image_folder(folder_path, detector, classifier, confidence_thresh=0.2, nms_thresh=0.4):
    
    # Make folder for saving detected images
    if not os.path.exists(os.path.join('./inference_detection', os.path.split(folder_path)[1])):
        os.mkdir(os.path.join('./inference_detection', os.path.split(folder_path)[1]))

    image_paths = sorted(glob2.glob(os.path.join(folder_path, '*.png')))
    info = []

    for image_path in image_paths:
        
        print("Processing image {}".format(image_path))
        # Load the image into program
        img = cv2.imread(image_path)

        # Make inference by detector
        print("\t--> [Stage 1] Localization ...")
        classes, scores, boxes = detector.detect(img, confidence_thresh, nms_thresh)

        # Crop the detect object regions    
        n_images = len(boxes)
        if n_images == 0:
            print("\tDon't detect any objects in image !")
            continue

        object_regions = [cv2.resize(img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]], 
                                    IMAGE_SIZE, cv2.INTER_AREA) for box in boxes]
        n_remain_imgs = BATCH_SIZE - (n_images % BATCH_SIZE)
        [object_regions.append(object_regions[0]) for _ in range(n_remain_imgs)]
        object_regions = np.array(object_regions)

        # Make inference by classifier
        print("\t--> [Stage 2] Classification ...")
        y_pred, x_recon = classifier.predict(object_regions)
        class_ids = np.argmax(y_pred[:len(y_pred)-n_remain_imgs, :], axis=1)
        scores = np.max(y_pred[:len(y_pred)-n_remain_imgs, :], axis=1)

        # Draw bounding boxes and corresponding classes.
        for (classid, score, box) in zip(class_ids, scores, boxes):
            label = "%s : %.3f" % (ENCODED2LABEL[classid], score)
            img = cv2.rectangle(img, box, COLOR, 2)
            #img = cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), COLOR, 1)
            img = cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

        info.append({'image_id':os.path.split(image_path)[1].split(".")[0], 'classes':class_ids, 
                      'scores':scores, 'boxes':boxes})
        saving_path = os.path.join('./inference_detection', os.path.split(folder_path)[1], 'detect_'+os.path.split(image_path)[1])
        cv2.imwrite(saving_path, img)
        print("\t--> {} inferenced and saved to {}".format(image_path, saving_path))

    return np.array(info)


def main(args):

    # Build detector model
    print("[INFO] Build the detector and load detect model from {} ...".format(args['detect_weights']))
    try:
        net = cv2.dnn.readNet(args['detect_weights'], args['config'], "darknet")
    except:
        print("[ERROR] Invalid detection model weights or config detection model file!")
        exit(0)    
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    detector = cv2.dnn_DetectionModel(net)
    detector.setInputParams(size=(512, 512), scale=1/255)

    # Build classifier
    print("[INFO] Build Capsule classifier and loading Capsule classifier from {} ...".format(args['classify_weights']))
    _, classifier, _ = CapsNet(input_shape=IMAGE_SIZE + (3,),
                    n_class=len(ENCODED2LABEL),
                    routings=ROUTINGS,
                    batch_size=BATCH_SIZE)
    if not os.path.exists(args['classify_weights']):
        print("[ERROR] Invalid clasisication model weight path !")
        exit(0)
    classifier.load_weights(args['classify_weights'])

    # Make inference progress
    if os.path.isfile(args['data_path']):
        inference_result = detect_single_image(args['data_path'], 
                        detector=detector, classifier=classifier, 
                        confidence_thresh=args['confidence_threshold'],
                        nms_thresh=args['nms_threshold'])
        
        #print(inference_result)
        saving_path = os.path.join('inference_detection', 'detect_'+ os.path.split(args['data_path'])[1])
        cv2.imwrite(saving_path, inference_result['detected_img'])
        print("[INFO] Saving the detected image into {}".format(saving_path))

    elif os.path.isdir(args['data_path']):
        inference_result = detect_image_folder(args['data_path'],
                        detector=detector, classifier=classifier, 
                        confidence_thresh=args['confidence_threshold'],
                        nms_thresh=args['nms_threshold'])
        
        # Saving file format: <confidence_threshold>_<nms_threshold>
        saving_file = os.path.join('./saving_info', '{}.npy'.format(os.path.split(args['data_path'])[1]))
        if not os.path.exists('./saving_info'):
            os.mkdir('./saving_info')
        np.save(saving_file, inference_result)
        print("Saved the detect information to {}".format(saving_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make inference on multi-images using yolo-v4')
    parser.add_argument('--data_path', '-path', type=str, default='./localization/test_image', 
                        help='The folder contains single image or image folder need inference.')
    parser.add_argument('--detect_weights', type=str, 
                        default='./localization/model_weights/custom-yolov4-detector_best.weights',
                        help='The path of detection model weights for inference.')
    parser.add_argument('--classify_weights', type=str, 
                        default='./classification/model_weights/weights-09.h5',
                        help='The path of classification model weights for inference.')
    parser.add_argument('--config', '-cfg', type=str, default='./localization/cfg/custom-yolov4-detector.cfg',
                        help='The configuration of yolo model.')
    parser.add_argument('--confidence_threshold', '-confidence', type=float, 
                        default=0.2, help='The confidence threshold for inference.')
    parser.add_argument('--nms_threshold', '-mns', type=float,
                        default=0.4, help='The Non-Maximal Suppression when inference.')
    args = vars(parser.parse_args())
    
    main(args)
    
    print("Finish detection !")