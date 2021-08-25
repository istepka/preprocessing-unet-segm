from data_generator import DataGenerator
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from data_loader import DataLoader
import random
import matplotlib.pyplot as plt
import utils
import time
import mlflow
from data_generator import DataGenerator
from sklearn.metrics import roc_auc_score

#SEEDS
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


#HYPERPARAMETERS
IMAGE_SIZE = 256
TRAIN_PATH = 'src/models/'
EPOCHS = 4
BATCH_SIZE = 8
DATASET_SIZE = 2500 #Number of datapoints 
FEATURE_CHANNELS = [32,64,128,256,512] #Number of feature channels at each floor of the UNet structure
DATA_AUGUMENTATION = None


class Trainer:

    def __init__(self) -> None:
        self.train_data = tuple()
        self.validation_data = tuple()
        self.model = None

        mlflow.tensorflow.autolog()
        mlflow.log_param('FEATURE_CHANNELS', FEATURE_CHANNELS)
        mlflow.log_param('IMAGE_SIZE', IMAGE_SIZE)
        mlflow.log_param('DATA_AUGUMENTATION', DATA_AUGUMENTATION)


    def load_data(self, n=DATASET_SIZE) -> None:
        raw_images, raw_masks = DataLoader().get_dataset(resolution=IMAGE_SIZE, n=n)
        norm_images, norm_masks = utils.normalize(raw_images, raw_masks)

        train_img, train_msk, test_img, test_msk = utils.split_train_test(norm_images, norm_masks)

        self.train_data = (train_img, train_msk)
        self.validation_data = (test_img, test_msk)

        print('Data loaded.')

    def build_model(self) -> str:
        self.model = UNet()

        optimizer = tf.keras.optimizers.Adam()

        self.model.compile(
            optimizer=optimizer, 
            loss='binary_crossentropy', 
            metrics=[
                'acc',
                tf.keras.metrics.AUC(),
                #auroc
                ]
            )
        
        print('Model built.')
        return self.model.summary()

    def train(self) -> None:
        #---------Train from directory --------------------------------------------------------------
        # with tf.device('/device:GPU:0'):
        #     train_x =  tf.convert_to_tensor(self.train_data[0], dtype=tf.float32)
        #     train_y = tf.convert_to_tensor(self.train_data[1], dtype=tf.float32)
        #     validation_x = tf.convert_to_tensor(self.validation_data[0], dtype=tf.float32) 
        #     validation_y = tf.convert_to_tensor(self.validation_data[1], dtype=tf.float32) 

        # self.model.fit(
        # x=train_x, 
        # y=train_y, 
        # batch_size=BATCH_SIZE , 
        # epochs=EPOCHS, 
        # validation_data=(validation_x, validation_y)
        # )
        #--------------------------------------------------------------------------------------------
        with tf.device('/device:GPU:0'):
            image_generator, mask_generator, valid_generator, valid_mask_generator = DataLoader().tf_get_generators(resolution=IMAGE_SIZE,batch_size=BATCH_SIZE)

            self.model.fit(
                x = zip(image_generator, mask_generator),
                epochs=EPOCHS,
                steps_per_epoch= len(image_generator),
                validation_data= zip(valid_generator, valid_mask_generator),
                validation_steps=len(valid_generator)
            )


        # -------- Traing with Generator (slower -> to be fixed) ------------------------------------
        # train_generator = DataGenerator('train', batch_size=BATCH_SIZE)                           
        # validation_generator = DataGenerator('validation', batch_size=BATCH_SIZE)                 

        # self.model.fit(
        #     x=train_generator,
        #     validation_data=validation_generator,
        #     epochs=EPOCHS,
        #     use_multiprocessing=True,
        #     workers=6
        # )
        #--------------------------------------------------------------------------------------------

      

    def save(self) -> None:
        timestamp = time.strftime(r"%d%m%Y-%H%M%S")
        path = f'{TRAIN_PATH}UNet_model_{IMAGE_SIZE}x{IMAGE_SIZE}_{timestamp}.h5'
        self.model.save_weights(path)

        print('Model weights saved: ' + path)

def auroc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

def down_block(x, filters, kernel_size=(3,3), padding="same", strides=1):
    with tf.device('/device:GPU:0'):
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(x)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(c)

        p = keras.layers.MaxPool2D((2,2), (2,2))(c)
    return c, p 

def up_block(x, skip, filters, kernel_size=(3,3), padding="same", strides=1):
    with tf.device('/device:GPU:0'):
        up_sampling = keras.layers.UpSampling2D((2,2))(x)
        concat = keras.layers.Concatenate()([up_sampling, skip])

        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(concat)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(c)

    return c

def bottleneck(x, filters, kernel_size=(3,3), padding="same", strides=1):
    with tf.device('/device:GPU:0'):
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(x)
        c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(c)

    return c

def jaccard_distance(y_true, y_pred, smooth=100):
    with tf.device('/device:GPU:0'):
        intersection = keras.backend.sum(keras.backend.abs(y_true * y_pred), axis=-1)
        sum_ = keras.backend.sum(keras.backend.square(y_true), axis = -1) + keras.backend.sum(keras.backend.square(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac)

def UNet():
    feature_maps = FEATURE_CHANNELS  #[64,128,256, 512, 1024]
    inputs = keras.layers.Input( (IMAGE_SIZE, IMAGE_SIZE, 1) )

    pool_0 = inputs
    conv_1, pool_1 = down_block(pool_0, feature_maps[0]) #512 -> 256
    conv_2, pool_2 = down_block(pool_1, feature_maps[1]) #256 -> 128 
    conv_3, pool_3 = down_block(pool_2, feature_maps[2]) #128 -> 64
    conv_4, pool_4 = down_block(pool_3, feature_maps[3]) #64 -> 32

    bn = bottleneck(pool_4, feature_maps[4])

    ups_1 = up_block(bn, conv_4, feature_maps[3]) #32 -> 64
    ups_2 = up_block(ups_1, conv_3, feature_maps[2]) #64 -> 128
    ups_3 = up_block(ups_2, conv_2, feature_maps[1]) #128 -> 256
    ups_4 = up_block(ups_3, conv_1, feature_maps[0]) #256 -> 512

    outputs = keras.layers.Conv2D(1, (1,1), padding='same', activation='sigmoid')(ups_4)

    model = keras.models.Model(inputs, outputs)
    return model



if __name__ == '__main__':
    tr = Trainer()
    #tr.load_data()
    tr.build_model()
    tr.train()
    tr.save()
