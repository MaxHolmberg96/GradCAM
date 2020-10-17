import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


from models.vgg16 import VGG16
from models.gradcam import GradCAM
from utils import *
from path import *


images_list = get_files(ILSVRC2012VAL_PATH)
boundingbox_list = get_files(ILSVRC2012VAL_BB_PATH)
image_index = 514


model = VGG16(weights="imagenet", classes=1000)
model.summary()
image = model.load_image(ILSVRC2012VAL_PATH + images_list[image_index])
image = model.preprocess_image(image)
image = tf.expand_dims(image, 0)
preds = model.predict(image)
print(model.decode_predictions(preds, top=5))

gradcam = GradCAM(model.model)
heatmaps, boundingboxes = get_heatmaps_and_bbs(gradcam=gradcam, image=image, predictions=preds, top=1)

format_ground_truth(ILSVRC2012VAL_BB_PATH + boundingbox_list[0])


