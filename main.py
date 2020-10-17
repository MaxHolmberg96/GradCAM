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
image_index = 48236


model = VGG16(weights="imagenet", classes=1000)
model.summary()
image = model.load_image(ILSVRC2012VAL_PATH + images_list[image_index])
image = model.preprocess_image(image)
image = tf.expand_dims(image, 0)
preds = model.predict(image)
decoded_preds = model.decode_predictions(preds, top=1)
print(decoded_preds)
gradcam = GradCAM(model.model)
heatmaps, boundingboxes = get_heatmaps_and_bbs(gradcam=gradcam, image=image, predictions=preds, top=1)

format_ground_truth(ILSVRC2012VAL_BB_PATH + boundingbox_list[0])

groundtruth = format_ground_truth(ILSVRC2012VAL_BB_PATH + boundingbox_list[image_index])
print(groundtruth)
print(evaluate(predictions=predictions, ground_truths=groundtruth))
show_image_with_bbs(image, [predictions[0][1], groundtruth[0][1]])
show_image_with_heatmap(gradcam, image, np.argmax(preds))