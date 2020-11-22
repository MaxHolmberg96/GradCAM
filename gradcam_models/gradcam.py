import numpy as np
import cv2


class gradcam:
    def __init__(self):
        self.model = None

    def predict(self, X):
        raise NotImplementedError

    def decode_predictions(self, preds, top):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError

    def preprocess_input(self, x):
        raise NotImplementedError

    def _get_last_conv_layer(self):
        conv_layer = "No convolutional layer found"
        for layer in self.model.layers:
            if "conv" in layer.name:
                conv_layer = layer.name
        return conv_layer

    def get_heatmap(self, c, data):
        raise NotImplementedError

    def _get_max_contour(self, heatmap, percentage):
        # resize heatmap and get contours
        # hetmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]), cv2.INTER_CUBIC)
        threshold = np.max(heatmap) * percentage
        heatmap[heatmap <= threshold] = 0
        heatmap[heatmap != 0] = 1
        heatmap = np.uint8(heatmap * 255)
        # thresh = cv2.threshold(heatmap, threshold, np.max(heatmap), cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # Find the largest contour
        max_contour = -1
        max_area = -1
        for c in cnts:
            area = cv2.contourArea(c)
            if area > max_area:
                max_area = area
                max_contour = c
        return max_area, max_contour

    def locate(self, image, decoded_preds, percentage=0.15, original_shape=None):
        raise NotImplementedError
