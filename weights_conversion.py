import torch
import torch.onnx
import torchvision
import torchvision.models as models
import sys

onnx_model_path = "Weights/conversion.onnx"

weights = 'yolov5s.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom',path=weights)
model.to('cpu')

model.eval()


dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True, opset_version=11)