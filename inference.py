import os
import traceback
import cv2

import pandas as pd
import numpy as np
from PIL import Image

import tensorflow as tf

import logger as Logger
logger = Logger.get_logger('hyper_fcn', './logs/inference')

def create_output_csv(image_paths, predictions, classes):

    output = {
            "file_name":[],
            "prediction":[],
            "score":[]
        }

    for i, preds in enumerate(predictions):

        output["file_name"].append(image_paths[i])
        output["prediction"].append(classes[np.argmax(preds)])
        output["score"].append(np.max(preds))

    return output

def construct_image_batch(image_group, BATCH_SIZE):
    """
        Pads the batch input with zeros for model prediction.
        Args
            image_group: np.array of shape (None, None, None, 3) or (None, 3, None, None)
            BATCH_SIZE: Batch size required to be fed into the model
        Returns
            Zero padded image_group with batches of size BATCH_SIZE
    """
    # Get the max image shape.
    max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

    # Construct an image batch object.
    image_batch = np.zeros((BATCH_SIZE,) + max_shape, dtype=tf.keras.backend.floatx())

    # Copy all images to the upper left part of the image batch object.
    for image_index, image in enumerate(image_group):
        image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

    if tf.keras.backend.image_data_format() == 'channels_first':
        image_batch = image_batch.transpose((0, 3, 1, 2))

    return image_batch

def resize_image(img, min_side_len=24):

    h, w, c = img.shape

    # limit the min side maintaining the aspect ratio
    if min(h, w) < min_side_len:
        im_scale = float(min_side_len) / h if h < w else float(min_side_len) / w
    else:
        im_scale = 1.

    new_h = int(h * im_scale)
    new_w = int(w * im_scale)

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return re_im, new_h / h, new_w / w

def preprocess_image(x):
    """ Preprocess an image by scaling pixels between -1 and 1, sample-wise.
    Args
        x: np.array of shape (None, None, 3) or (3, None, None)
    Returns
        The input with the pixels between -1 and 1.
    """

    # Covert always to float32 to keep compatibility with opencv.
    x = x.astype(np.float32)
    x /= 127.5
    x -= 1.

    x, rh, rw = resize_image(x)

    return x

def create_model_input(folder_path):
    """
        Creates numpy array of images in a given folder path
        Args
            folder_path: Path to folder where images are kept
        Returns
            A list of 3D numpy arrays where each list item is an image
    """
    image_paths = []
    images = []
    file_names = os.listdir(folder_path)
    for image_name in file_names:
            
        # Reading image the same way it is done while training
        img = np.asarray(Image.open(os.path.join(folder_path, image_name)).convert('RGB'))[:, :, ::-1]
    
        images.append(img)
        image_paths.append(image_name)
    return images, image_paths

def predict(folder_path, snapshot_dir):
    """
        Helper function to perform inference on the trained model.
        This function takes a folder with images as input. It then loads the images 
        and pre-processes the images similar to training. After successful pre-processing,
        the pre-trained model is loaded and inference is performed.
        The output is returned as a dataframe.
        Args
            folder_path: Path to folder where images are kept
            snapshot_dir: Path the snapshots folder where train_model.h5
                          and classes.txt are present
        Returns
            A dataframe with image name, model prediction and confidence score columns
    """
    logger.info('Reading input images')
    # Reading images from the input folder path
    images, image_paths = create_model_input(folder_path)

    logger.info('Processing input images')
    # Using the preprocessing function from the training generator
    images = list(map(preprocess_image, images))

    logger.info('Constructing image batches')
    # Constructing image batches similar to training generator
    images = construct_image_batch(images, len(images))

    logger.info('Loading model')
    # Loading pre-trained keras model
    model = tf.keras.models.load_model(os.path.join(snapshot_dir, 'train_model.h5'))

    logger.info('Getting model predictions')
    # Predicting on images batch
    predictions = model.predict(images)

    logger.info('Processing model output')
    # Reading class names
    with open(os.path.join(snapshot_dir, 'classes.txt')) as f:
        classes = f.read().splitlines()
    # Creating output dataframe
    output = create_output_csv(image_paths, predictions, classes)
    output = pd.DataFrame(output)
    
    return output


if __name__=="__main__":

    folder_path = "./test_images"
    snapshot_dir = "./snapshots"
    predictions = predict(folder_path, snapshot_dir)
    print(predictions)
