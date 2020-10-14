import tensorflow as tf
import numpy as np
from models.vgg16 import VGG16
from models.gradcam import GradCAM
import matplotlib.pyplot as plt
import cv2

vgg16_model = VGG16(weights="imagenet", classes=1000)
vgg16_model.summary()
image = vgg16_model.load_image("images/YellowLabradorLooking_new.jpg")
image = vgg16_model.preprocess_image(image)
image = tf.expand_dims(image, 0)
preds = vgg16_model.predict(image)

gradcam = GradCAM(vgg16_model.model)
layer_outputs = gradcam.get_all_layer_outputs(image)
result = gradcam(c=np.argmax(preds), conv_layer="block5_conv3", image=image)
cam = cv2.resize(result.numpy(), (224, 224), cv2.INTER_LINEAR)
plt.imshow(image[0, ...] / 255)
plt.imshow(cam, cmap="jet", alpha=0.5)
plt.show()
