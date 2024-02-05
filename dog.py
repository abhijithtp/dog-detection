from ultralytics import YOLO
import re
import pandas
# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on an image ,
#results = model('/content/cars.jpg')  # results list
results=model.predict('dog1.jpg', save=True, imgsz=320, conf=0.4,classes=[16],show_labels=True,show_boxes=True,show_conf=False)

#imgsz=320
