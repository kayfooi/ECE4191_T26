from ultralytics import YOLO

# # Load a YOLOv8n PyTorch model
# model = YOLO("YOLO_ball_detection.pt")

# # Export the model to NCNN format
# model.export(format="ncnn")  # creates 'yolov8n_ncnn_model'

# Load the exported NCNN model
ncnn_model = YOLO("./YOLO_ball_detection_ncnn_model")

# Run inference
results = ncnn_model("./test_imgs/test_images/testing0000.jpg")
print(results[0].boxes[0].xyxy)
results[0].save("result.jpg")