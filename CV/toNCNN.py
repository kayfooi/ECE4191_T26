"""
NCNN is a lighter neural network library that loads a lot faster than torch/ultralytics
Convert/test YOLO model here
"""
import time
import cv2
import numpy as np
import timeit

t = time.time()
from ultralytics import YOLO
e = time.time() - t
print(f'Ultralytics library loaded in {e} seconds.')

# Load the exported NCNN model
ncnn_model = YOLO("./YOLO_ball_detection_ncnn_model", task="detect")
# print(f'YOLO Model loaded in {time.time()-t-e} seconds.')

# # Run inference
t = time.time()
results = ncnn_model("./test_imgs/test_images/testing0000.jpg")
# print(results[0].boxes[0].xyxy)
e = time.time()
print(f'Ultralytics NCNN YOLO inferred in {e - t} seconds.')

# Load the OG YOLO model
ncnn_model = YOLO("YOLO_ball_detection.pt")

# # Run inference
t = time.time()
results = ncnn_model("./test_imgs/test_images/testing0000.jpg")
# print(results[0].boxes[0].xyxy)
e = time.time()
print(f'Ultralytics YOLO inferred in {e - t} seconds.')

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

img_proc = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

img = image.copy()
shape = img.shape[:2]  # current shape [height, width]
new_shape = (640, 640)

# Scale ratio (new / old)
r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

# Compute padding
stride = 32
ratio = r, r  # width, height ratios
new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

dw /= 2  # divide padding into 2 sides
dh /= 2

if shape[::-1] != new_unpad:  # resize
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
img = cv2.copyMakeBorder(
    img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
)  # add border

im = np.stack([img])
im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
in0 = np.ascontiguousarray(im/255).astype(np.float32)  # contiguous

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
        y = detection[1]
        y = (y - top) / (640 - top - bottom)
        
        center_x = int(xywh[0] * width)
        center_y = int(y * height)
        w = int(xywh[2] * width)
        h = int(xywh[3] * width)

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
        x, y, w, h = boxes[i]
        color = colors[class_ids[i]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y + 30), font, 2, color, 3)

e = time.time()
print(f'NCNN model load image and inferred in {e-t} seconds.')

# cv2.imshow('test',image)
# cv2.waitKey(0)
cv2.imwrite("result.jpg",image)

# # Load a YOLOv8n PyTorch model
# model = YOLO("YOLO_ball_detection.pt")

# # Export the model to NCNN format
# model.export(format="ncnn")  # creates 'yolov8n_ncnn_model'


# results[0].save("result.jpg")

# t = time.time()
# pt_model = YOLO("YOLO_ball_detection.pt")
# e = time.time()
# print(f'YOLO Model (pt) loaded in {t-e} seconds.')

# results = pt_model("./test_imgs/test_images/testing0000.jpg")
# print(results[0].boxes[0].xyxy)
# results[0].save("result.jpg")