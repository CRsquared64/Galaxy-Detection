import torch
import numpy
import time
import cv2 as cv


class GalaxyClassifier:

    def __init__(self, file_list, weights):
        self.file_list = file_list
        self.model = self.load_model(weights)
        self.classes = self.mode.names
        #check device
        if torch.cuda.is_avalible():
            self.device = 'cuda'
            print("GPU enabled")
        else:
            self.device = 'cpu'
            print("Using CPU")

        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
        return model
    def score(self, frame, x):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord, self.classes[int(x)]

    def draw(self, results, frame):
        labels, cord = results
        n = len(labels)
