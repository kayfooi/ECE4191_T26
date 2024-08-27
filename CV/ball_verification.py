import cv2
import numpy as np

def detect_yolo_tennis_ball_with_size_verification(frame):
    """
    Detects tennis balls using YOLOv3-tiny and verifies detections based on size.
    
    """
    # Load YOLOv3-tiny model
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    layer_names = net.getLayerNames()

    unconnected_out_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_out_layers[0], list) or isinstance(unconnected_out_layers[0], np.ndarray):
        output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
    else:
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

    classes = open("coco.names").read().strip().split("\n")

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

            if confidence > 0.3 and classes[class_id] == "sports ball":
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Size verification (you can tweak these values based on your setup)
                if 30 < w < 100 and 30 < h < 100:  # Expected size range for a tennis ball
                    centers.append((center_x, center_y))

    return centers