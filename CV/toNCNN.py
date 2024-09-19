import time
import cv2
import numpy as np

t = time.time()
from ultralytics import YOLO
e = time.time() - t
print(f'Ultralytics library loaded in {e} seconds.')

t = time.time()
import ncnn
e = time.time() - t
print(f'NCNN library loaded in {e} seconds.')

# Load the exported NCNN model
t = time.time()
net = ncnn.Net()
net.load_param("./YOLO_ball_detection_ncnn_model/model.ncnn.param")
net.load_model("./YOLO_ball_detection_ncnn_model/model.ncnn.bin")
e  = time.time()
print(f'NCNN model loaded in {e-t} seconds.')

t = time.time()
# Prepare input
image = cv2.imread("./test_imgs/test_images/testing0000.jpg")
height, width, _ = image.shape

img_proc = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255

img_proc = cv2.resize(img_proc, (640, 640))
in0 = np.empty((3, 640, 640)).astype(np.float32)
in0[0] = img_proc[:, :, 0]
in0[1] = img_proc[:, :, 1]
in0[2] = img_proc[:, :, 2]

# cv2.imshow('win', image)
# cv2.waitKey(0)

# Prepare input
mat_in = ncnn.Mat(in0)

# Run inference
extractor = net.create_extractor()
extractor.input("in0", mat_in)
ret, mat_out = extractor.extract("out0")
out = np.array(mat_out)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []

max = np.argmax(out[4, :])
print(out[:, max])

for i in range(mat_out.w):
    detection = out[:, i]
    scores = detection[4:]
    class_id = np.argmax(scores)
    confidence = scores[class_id]
    if confidence > 0.5:
        # Object detected
        xywh = detection[:4] / 640
        center_x = int(xywh[0] * width)
        center_y = int(xywh[1] * height)
        w = int(xywh[2] * width)
        h = int(xywh[3] * height)

        # Rectangle coordinates
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)

        boxes.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)
        

indexes = cv2.dnn.NMSBoxes(boxes, confidences,score_threshold=0.1,nms_threshold=0.1,top_k=5)
print(len(boxes), len(indexes))
classes = ['tennis-ball']
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))
labels = ['tennis-ball']
for i in range(len(boxes)):
    if i in indexes:
        label = str(classes[class_ids[i]])
        if label in labels:
            print(label, boxes[i])
        # x, y, w, h = boxes[i]
        # color = colors[class_ids[i]]
        # cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        # cv2.putText(image, label, (x, y + 30), font, 2, color, 3)

# cv2.imshow('test',image)
# cv2.waitKey(0)

e = time.time()
print(f'NCNN model load image and inferred in {e-t} seconds.')




# # Load a YOLOv8n PyTorch model
# model = YOLO("YOLO_ball_detection.pt")

# # Export the model to NCNN format
# model.export(format="ncnn")  # creates 'yolov8n_ncnn_model'

# Load the exported NCNN model
t = time.time()
ncnn_model = YOLO("./YOLO_ball_detection_ncnn_model", task="detect")
e = time.time()
print(f'YOLO ncnn Model loaded w/ ultralytics in {t-e} seconds.')

# Run inference
results = ncnn_model("./test_imgs/test_images/testing0000.jpg")
print(results)
# results[0].save("result.jpg")

# t = time.time()
# pt_model = YOLO("YOLO_ball_detection.pt")
# e = time.time()
# print(f'YOLO Model (pt) loaded in {t-e} seconds.')

# results = pt_model("./test_imgs/test_images/testing0000.jpg")
# print(results[0].boxes[0].xyxy)
# results[0].save("result.jpg")