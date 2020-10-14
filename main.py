import tensorflow as tf
import numpy as np
from models.vgg19 import VGG19
from models.gradcam import GradCAM

VGG19_model = VGG19(weights="imagenet", classes=1000)
VGG19_model.summary()
image = VGG19_model.load_image("images/YellowLabradorLooking_new.jpg")
image = VGG19_model.preprocess_image(image)
image = tf.expand_dims(image, 0)
preds = VGG19_model.predict(image)
VGG19_model.summary()
print(VGG19_model.decode_predictions(preds))

gradcam = GradCAM(VGG19_model.model)
heatmap = gradcam.get_heatmap(c=np.argmax(preds), image=image)
gradcam.show(heatmap, image)

