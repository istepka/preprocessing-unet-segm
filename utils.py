from typing import Any, Tuple
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from preprocessing.preprocessor import Preprocessor
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import matplotlib.pyplot as plt

#PREPROCESSING
def normalize(images, masks):
    images = images / 255
    masks = (masks > 0).astype(float)
    return images, masks

def norm_per_channel(images, mean=None) -> Tuple[Any, float]:
    if mean is None:
        mean = images.mean()
    return images - mean, mean

def apply_gaussian_blur(images, filter_radius=2) -> Any:
    resolution = images[0].shape[0]
    for i, im in enumerate(images):
        p = Preprocessor(Image.fromarray(im.flatten().reshape((resolution,resolution))).convert(mode='L'))
        p.apply_gaussian_blur(filter_radius)
        images[i] =  p.get_processed_np_img(normalized=False)
    return images

def apply_histogram_equalization(images, cutoff_percentage) -> Any:
    resolution = images[0].shape[0]
    for i, im in enumerate(images):
        p = Preprocessor(Image.fromarray(im.flatten().reshape((resolution,resolution))).convert(mode='L'))
        p.hist_enchance_contrast(cutoff_percentage)
        images[i] =  p.get_processed_np_img(normalized=False)
    return images

def apply_zca_normalization(images, fit_dataset=None) -> Any:
    gen = ImageDataGenerator(zca_whitening=True)

    if fit_dataset is None:
        gen.fit(images, seed=133)
    else:
        gen.fit(fit_dataset, seed=133)

    for i, im in enumerate(images):
        images[i] = gen.standardize(im)

    return images

#FUNCTIONS
def iou(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.bool)
    y_true = tf.cast(y_true, tf.bool)
    intersection = tf.math.logical_and(y_true, y_pred)
    union = tf.math.logical_or(y_true, y_pred)
    iou_score = tf.math.reduce_sum(tf.cast(intersection, tf.int16)) / tf.math.reduce_sum(tf.cast(union, tf.int16))
    return iou_score

def auroc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

def jaccard_index(y_true, y_pred, smooth=0.0001):
    with tf.device('/device:GPU:0'):
        y_pred = tf.math.greater_equal(y_pred, 0.5)
        y_true = tf.cast(y_true, tf.bool)
        intersection = tf.math.reduce_sum( tf.cast(tf.math.logical_and(y_pred, y_true), tf.int8))
        union = tf.math.reduce_sum( tf.cast(tf.math.logical_or(y_pred, y_true), tf.int8))

        jac_index = intersection /  union
        
    return jac_index

def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

#UTILITIES
def display(images, rows=1):
    fig = plt.figure(figsize=(8, 8))
    columns = len(images)

    for i in range(1, len(images)+1):
        fig.add_subplot(rows, columns, i)

        if len(images[i-1].shape) > 3: 
            images[i-1] = np.reshape(images[i-1], (256,256,1))
        
        plt.imshow(images[i-1], cmap='gray')
        
    plt.show()
    
