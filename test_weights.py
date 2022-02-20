import cv2
import torch
import numpy
import time
import cv2 as cv
import platform


class GalaxyClassifier:

    def __init__(self, file_list, weights):
        self.file_list = file_list
        self.model = self.load_model(weights)
        self.classes = self.model.names

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
        labels, cord = results
        n = len(labels)
        x_shape = frame.shape[1]
        y_shape = frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1 = int(row[0]*x_shape)
                y1 = int(row[1]*y_shape)
                x2 = int(row[2]*x_shape)
                y2 = int(row[3]*y_shape)
                bgr = (0,255,0)
                cv.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)
                cv.putText(frame, self.class_convert(labels[i]), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.9,(118, 185, 0))
        return frame
    def __call__(self):
        img = self.get_img()

        while True:
            results = self.score(img)
            img = self.draw(results, img)
            cv.imwrite("Detected.jpg", img)

            if cv2.waitKey(5) & 0xFF ==27:
                break


if __name__ == '__main__':
    galaxyClassifier = GalaxyClassifier(file_list='100090.jpg', weights='we.pt')
    galaxyClassifier()

