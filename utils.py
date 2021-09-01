import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#import wmi
from sklearn.metrics import roc_auc_score

def split_train_test(images, masks, validation_split=0.8):
    split_index = int(images.shape[0] * 0.8)
    return images[0:split_index], masks[0:split_index], images[split_index:], masks[split_index:]

def normalize( images, masks):
    images = images / 255
    masks = (masks > 0).astype(float)
    return images, masks

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


