import torch
import torch.onnx
import torchvision
import torchvision.models as models
import sys
import platform


class ConvertWeights:

    def __init__(self, weights, output):
        self.model = self.load_model(weights)
        self.output = output

    def device(self):

        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            print(f"Using CPU: {platform.processor()}")

    def load_model(self, weights):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
        return model

    def __call__(self, *args, **kwargs):
        self.model.to(self.device())
        self.model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(self.model, dummy_input, self.output, verbose=True, opset_version=11)


if __name__ == '__main__':
    convertWeights = ConvertWeights(weights='Weights/TestWeights.pt', output='Weights/conversion.onnx')
    convertWeights()
