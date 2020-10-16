import tensorflow as tf
import numpy as np
from models.vgg16 import VGG16
from models.gradcam import GradCAM
from utils import draw_bounding_box_from_file, get_files, ILSVRC2012VAL_BB_PATH, ILSVRC2012VAL_PATH


images_list = get_files(ILSVRC2012VAL_PATH)
boundingbox_list = get_files(ILSVRC2012VAL_BB_PATH)

model = VGG16(weights="imagenet", classes=1000)
model.summary()
image = model.load_image(ILSVRC2012VAL_PATH + images_list[0])
image = model.preprocess_image(image)
image = tf.expand_dims(image, 0)
preds = model.predict(image)
print(preds.shape)
model.summary()
print(model.decode_predictions(preds, top=1))

gradcam = GradCAM(model.model)
heatmap = gradcam.get_heatmap(c=np.argmax(preds), image=image).numpy()
threshold = (0.15 * np.max(heatmap))
binarized_heatmap = np.zeros_like(heatmap)
binarized_heatmap[heatmap > threshold] = 1
#gradcam.show(binarized_heatmap, image)

original = model.load_image(ILSVRC2012VAL_PATH + images_list[0])
original = tf.expand_dims(original, 0)
original = draw_bounding_box_from_file(original, boundingbox_list[0])
import matplotlib.pyplot as plt
plt.imshow(original.numpy()[0, ...] / 255)
plt.show()
#gradcam.show(heatmap, image)


