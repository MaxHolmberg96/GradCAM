import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
import synset_mappings


class GradCAM:
    def __init__(self, model):
        self.model = model

    def _get_last_conv_layer(self):
        conv_layer = "No convolutional layer found"
        for layer in self.model.layers:
            if "conv" in layer.name:
                conv_layer = layer.name
        return conv_layer

    def get_all_layer_outputs(self, X):
        func = tf.keras.backend.function(
            [self.model.layers[0].input],
            [layer.output for layer in self.model.layers[1:]],
        )
        layer_outputs = func([X])
        layer_outputs.insert(0, X)  # Add so image is the output of the input :)
        return layer_outputs

    def get_gradients(self, c, image):
        conv_layer = self._get_last_conv_layer()
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(conv_layer).output, self.model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            y_c = predictions[:, c]

        conv_output = conv_outputs[0]
        """heatmaps = conv_output.numpy()
        for i in range(heatmaps.shape[-1]):
            heatmap = heatmaps[..., i]
            heatmapshow = None
            heatmap = cv2.resize(heatmap, (535, 431))
            heatmapshow = cv2.normalize(
                heatmap, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
            cv2.imwrite("activations/activation_{}.png".format(i), heatmapshow)
        """
        grads = tape.gradient(y_c, conv_outputs)
        grads = grads[0]
        return grads, conv_output

    def get_gradients_text(self, c, image, conv_layer, inverse=False):
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(conv_layer).output, self.model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            y_c = predictions[:, c]

        conv_output = conv_outputs[0]
        grads = tape.gradient(y_c, conv_outputs)
        grads = grads[0]
        return grads, conv_output

    def get_heatmap(self, c, image):
        grads, output = self.get_gradients(c, image)
        alphas = tf.reduce_mean(grads, axis=[0, 1])
        linear_combinaton = tf.reduce_sum(alphas * output, axis=-1)
        linear_combinaton = cv2.resize(
            linear_combinaton.numpy(), (image.shape[2], image.shape[1]), interpolation=cv2.INTER_CUBIC
        )
        cam = tf.nn.relu(linear_combinaton).numpy()
        cam = cam / np.max(cam)
        return cam

    def get_heatmap_text(self, c, image, layer_name, inverse=False):
        grads, output = self.get_gradients_text(c, image, layer_name, inverse)
        alphas = tf.reduce_mean(grads, axis=[0, 1])
        linear_combinaton = tf.reduce_sum(alphas * output, axis=-1)
        linear_combinaton = cv2.resize(
            linear_combinaton.numpy(), (image.shape[0], image.shape[1]), interpolation=cv2.INTER_CUBIC
        )
        cam = tf.nn.relu(linear_combinaton).numpy()
        cam = cam / np.max(cam)
        return cam

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
        if image.ndim != 3:
            raise Exception("Image should be of shape (width, height, channels)")
        image = tf.expand_dims(image, 0)
        boundingboxes = []
        for name, _, _ in decoded_preds:
            heatmap = self.get_heatmap(c=synset_mappings.name_to_index[name]["index"], image=image)
            max_area, max_contour = self._get_max_contour(heatmap, percentage)
            # img = np.uint8(image[0].numpy())
            # cv2.drawContours(img, [max_contour], -1, (0, 255, 0), 1)
            # plt.imshow(img)
            # plt.imshow(heatmap, alpha=0.3)
            # plt.show()
            # plt.imshow(heatmap)
            # plt.show()
            if max_area != -1:
                xmin, ymin, w, h = cv2.boundingRect(max_contour)
                xmax = xmin + w
                ymax = ymin + h
                # xmin *= 224 / 14
                # ymin *= 224 / 14
                # xmax *= 224 / 14
                # ymax *= 224 / 14
                if original_shape is not None:
                    height, width, _ = original_shape
                    # new_height = height * 256 // min(original_shape[:2])
                    # new_width = width * 256 // min(original_shape[:2])
                    # startx = new_width // 2 - (224 // 2)
                    # starty = new_height // 2 - (224 // 2)
                    # xmin += startx
                    # xmax += startx
                    # ymin += starty
                    # ymax += starty

                    # xmin *= original_shape[1] / new_width
                    # xmax *= original_shape[1] / new_width
                    # ymin *= original_shape[0] / new_height
                    # ymax *= original_shape[0] / new_height

                    xmin *= original_shape[1] / image.shape[2]
                    xmax *= original_shape[1] / image.shape[2]
                    ymin *= original_shape[0] / image.shape[1]
                    ymax *= original_shape[0] / image.shape[1]

                boundingboxes.append(
                    (
                        name,
                        {
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax,
                        },
                    )
                )
            else:
                boundingboxes.append((name, None))
        return boundingboxes
