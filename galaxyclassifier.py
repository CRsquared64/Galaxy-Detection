import cv2
import torch
import numpy
import time
import cv2 as cv
import platform

threshold = ""
default_threshold = '0.25'

if not threshold:
    threshold = default_threshold
    print("Default threshold active.")


class GalaxyClassifier:

    def __init__(self, file_list, weights, thresh):
        self.file_list = file_list
        self.model = self.load_model(weights)
        self.classes = self.model.names
        self.thresh = thresh

    def load_model(self, weights):
        # check device
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            print(f"Using CPU: {platform.processor()}")

        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)
        return model

    def get_img(self):
        w = 412
        h = 412
        # reminder to self, put any modifications here Eg. stretch to 412

        return cv2.imread(self.file_list)

    def score(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_convert(self, x):
        return self.classes[int(x)]

    def draw(self, results, frame):
        thresh=float(self.thresh)
        labels, cord = results
        n = len(labels)
        x_shape = frame.shape[1]
        y_shape = frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= thresh:
                thresh_text = str(round(float(row[4]), 2))
                print(f"Detected galaxy with confidence {thresh_text}: {self.class_convert(labels[i])} ")
                x1 = int(row[0] * x_shape)
                y1 = int(row[1] * y_shape)
                x2 = int(row[2] * x_shape)
                y2 = int(row[3] * y_shape)
                bgr = (0, 255, 0)
                cv.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv.putText(frame, (self.class_convert(labels[i]) + " " + thresh_text), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.9, (118, 185, 0))
        return frame

    def __call__(self):
        img = self.get_img()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.score(img)
        img = self.draw(results, img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite("yolo.jpg", img)


if __name__ == '__main__':
    galaxyClassifier = GalaxyClassifier(file_list='999622.jpg', weights='Weights/yolov5s.pt', thresh=threshold)
    galaxyClassifier()
