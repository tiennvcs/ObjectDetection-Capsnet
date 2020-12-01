import cv2
from classification.config import ENCODED2LABEL, BATCH_SIZE, IMAGE_SIZE, ROUTINGS


def detect_single_image(image_path, detector, classifier, confidence_thresh=0.2, nms_thresh=0.4):

    # Load image into program
    if not os.path.exists(image_path):
        print("The {} is INVALID path !".format(image_path))
        exit(0)

    # Load image into program
    img = cv2.imread(image_path)

    start = time.time()
    # Make inference
    classes, scores, boxes = model.detect(img, confidence_thresh, nms_thresh)

    # Draw bounding boxes and corresponding classes.
    object_regions = np.array([img[box[0]:box[2], box[1]:box[3], :] for box in boxes])
    print(object_regions.shape)

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        #label = "%s : %f" % (class_names[classid[0]], score)

        img = cv2.rectangle(img, box, color, 1)
        img = cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return {'classes':classes, 'scores':scores, 'boxes':boxes, 'detected_img':img}


def detect_image_folder(folder_path, model, class_names, confidence_thresh=0.2, nms_thresh=0.4):
    
    # Make folder for saving detected images
    if not os.path.exists(os.path.join('./inference_images', os.path.split(folder_path)[1])):
        os.mkdir(os.path.join('./inference_images', os.path.split(folder_path)[1]))

    image_paths = sorted(glob2.glob(os.path.join(folder_path, '*.png')))
    info = []

    for image_path in image_paths:
        
        # Load the image into program
        img = cv2.imread(image_path)

        # Make inference
        classes, scores, boxes = model.detect(img, confidence_thresh, nms_thresh)

        # Draw bounding boxes and corresponding classes.
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = r"%s : %f" % (class_names[classid[0]], score)
            img = cv2.rectangle(img, box, color, 2)
            img = cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        info.append({'image_id':os.path.split(image_path)[1].split(".")[0], 'classes':classes, 
                      'scores':scores, 'boxes':boxes})
        saving_path = os.path.join('./inference_images', os.path.split(folder_path)[1], 'detect_'+os.path.split(image_path)[1])
        cv2.imwrite(saving_path, img)
        print("{} inferenced and saved to {}".format(image_path, saving_path))

    return np.array(info)


def main(args):

    # Build detector model
    try:
        net = cv2.dnn.readNet(args['model_weights'], args['config'], "darknet")
    except:
        print("Invalid model weights or config model file!")
        exit(0)    
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    detector = cv2.dnn_DetectionModel(net)
    detector.setInputParams(size=(512, 512), scale=1/255)

    # Build classifier
    _, classifier, _ = CapsNet(input_shape=imgs.shape[1:],
                    n_class=len(ENCODED2LABEL),
                    routings=ROUTINGS,
                    batch_size=BATCH_SIZE)

    print("[INFO] Loading model weights from {}".format(args['weights']))
    classifier.load_weights(args['weights'])

    if os.path.isfile(args['data_path']):
        inference_result = detect_single_image(args['data_path'], 
                        detector=detector, classifier=classifier, 
                        confidence_thresh=args['confidence_threshold'],
                        nms_thresh=args['nms_threshold'])
        
        #print(inference_result)
        saving_path = os.path.join('inference_images', 'detect_'+os.path.split(args['data_path'])[1])
        cv2.imwrite(saving_path, inference_result['detected_img'])
        print("Saving the detected image into {}".format(saving_path))

    elif os.path.isdir(args['data_path']):
        inference_result = detect_image_folder(args['data_path'],
                        model=model, class_names=class_names, 
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
    parser.add_argument('--data_path', '-path', type=str, default='./localization/test_data', 
                        help='The folder contains single image or image folder need inference.')
    parser.add_argument('--detect_weights', type=str, 
                        default='./localization/model_weights/custom-yolov4-detector_best.weights',
                        help='The path of model weights for inference.')
    parser.add_argument('--config', '-cfg', type=str, default='localization/cfg/custom-yolov4-detector.cfg',
                        help='The configuration of yolo model.')
    parser.add_argument('--confidence_threshold', '-confidence', type=float, 
                        default=0.2, help='The confidence threshold for inference.')
    parser.add_argument('--nms_threshold', '-mns', type=float,
                        default=0.4, help='The Non-Maximal Suppression when inference.')
    args = vars(parser.parse_args())
    
    main(args)
    
    print("Finish detection !")