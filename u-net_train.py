import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
#from tensorflow.keras.layers.experimental import preprocessing
#from tensorflow_examples.models.pix2pix import pix2pix
#from IPython.display import clear_output
import matplotlib.pyplot as plt
from data_loader import DataLoader
import numpy as np

import tensorflow_datasets as tfds



def normalize(input_image, input_mask):
    '''Normalize color values to be between 0-1'''
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask

def load_dataset():
    '''Get dataset.\n
    Returns:\n
    `tuple`: train_x, train_y, test_x, test_y'''

    raw_imgs, raw_masks = DataLoader().get_dataset()
    imgs = list(map( lambda x: tf.convert_to_tensor(x), raw_imgs ))
    masks = list(map( lambda x: tf.convert_to_tensor(x), raw_masks ))
    split_index = int(len(imgs) * 0.75)

    return ( imgs[0:split_index], masks[0:split_index], imgs[split_index:], masks[split_index:] )



train_x, train_y, test_x, test_y = load_dataset()

TRAIN_LENGTH = len(train_x)
BATCH_SIZE = 2
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
            plt.axis('off')
    plt.show()


print(train_x[0].shape)
