import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
print(model.names)
