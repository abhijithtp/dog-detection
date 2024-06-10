from ultralytics import YOLO
import re
import pandas
# Load a pretrained YOLOv8n model
model = YOLO('yolov5n.pt')

# Run inference on an image ,
#results=model.predict('dog1.jpg', save=True, imgsz=320, conf=0.4,classes=[16],show_labels=True,show_boxes=True,show_conf=False)
freeze=10
freeze = [f"model.{x}." for x in range(freeze)]  # layers to freeze
for k, v in model.named_parameters():
    v.requires_grad = True  # train all layers
    if any(x in k for x in freeze):
        print(f"freezing {k}")
        v.requires_grad = False
train.py --batch 48 --weights yolov5m.pt --data voc.yaml --epochs 50 --cache --img 512 --hyp hyp.finetune.yaml
#imgsz=320
