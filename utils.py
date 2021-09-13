from typing import Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.type_check import imag
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from preprocessing.preprocessor import Preprocessor
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

#PREPROCESSING
def split_train_test(images, masks, validation_split=0.8):
    # np.random.seed(1)
    # mat = np.random.choice(a=[False, True], size=(len(images)), p=[validation_split, 1-validation_split])
    # return images[~mat], masks[~mat], images[mat], masks[mat]
    idx = int(len(images) * validation_split)
    return images[0:idx], masks[0:idx], images[idx:], masks[idx:]

def normalize( images, masks):
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

#DISPLAY
def display_pair(image1, image2, title1='', title2=''):
   
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1,2,1)
    ax.set_title(title1)
    ax.imshow(image1, cmap='gray')

    ax1 = fig.add_subplot(1,2,2)
    ax1.set_title(title2)
    ax1.imshow(image2, cmap='gray')


    plt.show()

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

def jaccard_distance(y_true, y_pred, smooth=100):
    with tf.device('/device:GPU:0'):
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=-1)
        sum_ = tf.keras.backend.sum(tf.keras.backend.square(y_true), axis = -1) + tf.keras.backend.sum(tf.keras.backend.square(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac)

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

if __name__ == '__main__':

    inp = tf.convert_to_tensor([0.1, 0.2, 0.6, 0.8], dtype=tf.float32)
    tar = tf.convert_to_tensor([1, 0, 1, 1], dtype=tf.float32)

    j = jaccard_index(inp, tar)

    print('Jaccard index: ', j)

  
    

    
