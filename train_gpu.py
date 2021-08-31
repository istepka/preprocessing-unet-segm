from data_generator import DataGenerator
import numpy as np
import tensorflow as tf 
from data_loader import DataLoader
import random
import matplotlib.pyplot as plt
import utils
import time
import mlflow
from data_generator import DataGenerator
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

#SEEDS
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


#HYPERPARAMETERS
IMAGE_SIZE = 256
TRAIN_PATH = 'src/models/'
EPOCHS = 2
BATCH_SIZE = 16
DATASET_SIZE = 2500 #Number of datapoints 
FEATURE_CHANNELS = [32,64,128,256,512] #Number of feature channels at each floor of the UNet structure
DATA_AUGUMENTATION = True

#PARAMETERS
params = {
    'shuffle': True,
    'seed': 1,
    'featurewise_center': True
}
augument = {
    'featurewise_center': True,
    #'featurewise_std_normalization': True,
    'rotation_range': 15,
    'zoom_range': 0.15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip':True,
    'vertical_flip': True,
    'shear_range': 0.1
}
augument_mask = {
    #'featurewise_center': True,
    #'featurewise_std_normalization': True,
    'rotation_range': 10,
    'zoom_range': 0.1,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip':True,
    'vertical_flip': True,
    'shear_range': 0.05
}


class Trainer:

    def __init__(self) -> None:
        self.train_data = tuple()
        self.validation_data = tuple()
        self.model = None

        mlflow.tensorflow.autolog()
        mlflow.log_param('FEATURE_CHANNELS', FEATURE_CHANNELS)
        mlflow.log_param('IMAGE_SIZE', IMAGE_SIZE)
        mlflow.log_param('DATA_AUGUMENTATION', DATA_AUGUMENTATION)
        mlflow.log_param('BATCH_SIZE', BATCH_SIZE)
        mlflow.log_params(params)
        mlflow.log_params(augument)

    def load_data(self, n=DATASET_SIZE) -> None:
        raw_images, raw_masks = DataLoader().get_dataset(resolution=IMAGE_SIZE, n=n)
        norm_images, norm_masks = utils.normalize(raw_images, raw_masks)

        train_img, train_msk, test_img, test_msk = utils.split_train_test(norm_images, norm_masks)

        self.train_data = (train_img, train_msk)
        self.validation_data = (test_img, test_msk)

        print('Data loaded.')

    def quick_load_data(self):
        with tf.device('/device:GPU:0'):
            imgs, msks = DataLoader().get_data_from_npy('data.npy')

            norm_images, norm_masks = utils.normalize(imgs, msks)

            train_img, train_msk, test_img, test_msk = utils.split_train_test(norm_images, norm_masks)
            #train_img, train_msk, test_img, test_msk = utils.split_train_test(imgs, msks)

            if DATA_AUGUMENTATION:
                print('Data augumentation on')
            datagen = ImageDataGenerator(**augument)
            datagen.fit(norm_images, seed=params['seed'])

            maskdatagen = ImageDataGenerator(**augument_mask)
            maskdatagen.fit(norm_masks,seed=params['seed'])

            testdatagen = ImageDataGenerator(featurewise_center=augument['featurewise_center'])
            testdatagen.fit(norm_images, seed=params['seed'])

            self.image_iterator = datagen.flow(train_img, batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
            self.mask_iterator = maskdatagen.flow(train_msk, batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
            self.train_iterator = zip(self.image_iterator, self.mask_iterator)

            self.test_image_iterator = testdatagen.flow(test_img, batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
            self.test_mask_iterator = testdatagen.flow(test_msk, batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
            self.test_iterator = zip(self.test_image_iterator, self.test_mask_iterator)

           

            print('Data loaded.')
            print(f'Training samples: {len(train_img)}, mean: {np.mean(train_img)},\n \
                Validation samples: {len(test_img)}, mean: {np.mean(test_img)}')


    def build_model(self) -> str:
        self.model = UNet()

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        self.model.compile(
            optimizer=optimizer, 
            loss='binary_crossentropy', 
            metrics=[
                'acc',
                tf.keras.metrics.AUC(),
                #iou,
                tf.keras.metrics.MeanIoU(num_classes=2),
                tf.keras.metrics.Precision(),
                #tf.keras.metrics.TrueNegatives(),
                #tf.keras.metrics.TruePositives()
                #auroc
                ]

            )
        
        print('Model built.')
        return self.model.summary()

    def train(self) -> None:
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)


        with tf.device('/device:GPU:0'):
            self.model.fit(
                self.train_iterator,
                steps_per_epoch=len(self.image_iterator),
                epochs=EPOCHS, 
                validation_steps=len(self.test_image_iterator),
                validation_data=self.test_iterator,
                callbacks=[early_stopping]
            )


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
        
        #-------------------Train from flow--------------------------------------------------------------
        # with tf.device('/device:GPU:0'):
        #     image_generator, mask_generator, valid_generator, valid_mask_generator = DataLoader().tf_get_generators(resolution=IMAGE_SIZE,batch_size=BATCH_SIZE)

        #     self.model.fit(
        #         x = zip(image_generator, mask_generator),
        #         epochs=EPOCHS,
        #         steps_per_epoch= len(image_generator),
        #         validation_data= zip(valid_generator, valid_mask_generator),
        #         validation_steps=len(valid_generator)
        #     )
        #---------------------------------------------------------------------------------------


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

def iou(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.bool)
    y_true = tf.cast(y_true, tf.bool)
    intersection = tf.math.logical_and(y_true, y_pred)
    union = tf.math.logical_or(y_true, y_pred)
    iou_score = tf.math.reduce_sum(tf.cast(intersection, tf.int16)) / tf.math.reduce_sum(tf.cast(union, tf.int16))
    return iou_score

def auroc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

def down_block(x, filters, kernel_size=(3,3), padding="same", strides=1):
    with tf.device('/device:GPU:0'):
        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(x)
        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(c)

        p = tf.keras.layers.MaxPool2D((2,2), (2,2))(c)
    return c, p 

def up_block(x, skip, filters, kernel_size=(3,3), padding="same", strides=1):
    with tf.device('/device:GPU:0'):
        up_sampling = tf.keras.layers.UpSampling2D((2,2))(x)
        concat = tf.keras.layers.Concatenate()([up_sampling, skip])

        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(concat)
        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(c)

    return c

def bottleneck(x, filters, kernel_size=(3,3), padding="same", strides=1):
    with tf.device('/device:GPU:0'):
        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(x)
        c = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation='relu')(c)

    return c

def jaccard_distance(y_true, y_pred, smooth=100):
    with tf.device('/device:GPU:0'):
        intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=-1)
        sum_ = tf.keras.backend.sum(tf.keras.backend.square(y_true), axis = -1) + tf.keras.backend.sum(tf.keras.backend.square(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac)

def UNet():
    feature_maps = FEATURE_CHANNELS  #[64,128,256, 512, 1024]
    inputs = tf.keras.layers.Input( (IMAGE_SIZE, IMAGE_SIZE, 1) )

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

    outputs = tf.keras.layers.Conv2D(1, (1,1), padding='same', activation='sigmoid')(ups_4)

    model = tf.keras.models.Model(inputs, outputs)
    return model



if __name__ == '__main__':
    #utils.display_sys_info()
    tr = Trainer()
    #tr.load_data()
    tr.quick_load_data()
    tr.build_model()
    tr.train()
    tr.save()

