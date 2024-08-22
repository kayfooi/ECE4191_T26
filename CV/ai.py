import cv2
import numpy as np

def detect_yolo_centers(frame):
    """
    Detects objects using YOLOv3-tiny and returns their center coordinates.

    :param frame: Input image/frame (BGR format).
    :return: A list of center coordinates [(x1, y1), (x2, y2), ...].
    """
    # Load YOLOv3-tiny model
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    layer_names = net.getLayerNames()

    # Handle different versions of OpenCV
    unconnected_out_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_out_layers[0], list) or isinstance(unconnected_out_layers[0], np.ndarray):
        output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
    else:
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

    classes = open("coco.names").read().strip().split("\n")

    # Initialize video stream
    height, width, _ = frame.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the detections
    centers = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.4 and classes[class_id] == "sports ball":  # Replace "mouse" with "tennis ball" if available
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Append center coordinates
                centers.append((center_x, center_y))

    return centers