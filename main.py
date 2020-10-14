import tensorflow as tf
from models.vgg16 import VGG16

vgg16_model = VGG16(weights="imagenet", classes=1000)
image = vgg16_model.load_image("images/YellowLabradorLooking_new.jpg")
image = vgg16_model.preprocess_image(image)
preds = vgg16_model.predict(tf.expand_dims(image, 0))
print(vgg16_model.decode_predictions(preds))