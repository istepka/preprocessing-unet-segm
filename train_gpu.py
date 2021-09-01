import numpy as np
import tensorflow as tf 
from data_loader import DataLoader
import random
import utils
import time
import mlflow
from data_generator import DataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from UNet import UNet

#SEEDS
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


#HYPERPARAMETERS
IMAGE_SIZE = 256
TRAIN_PATH = 'src/models/'
EPOCHS = 100
BATCH_SIZE = 8
DATASET_SIZE = 2500 #Number of datapoints 
FEATURE_CHANNELS = [64,128,256,512, 1024] #Number of feature channels at each floor of the UNet structure
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
    'zoom_range': 0.1,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip':True,
    'vertical_flip': True,
    #'shear_range': 0.1
}
augument_mask = {
    #'featurewise_center': True,
    #'featurewise_std_normalization': True,
    'rotation_range': 15,
    'zoom_range': 0.1,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip':True,
    'vertical_flip': True,
    #'shear_range': 0.1
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

    def quick_load_data(self):
        '''Load data from numpy table file'''
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
            testmaskdatagen = ImageDataGenerator(featurewise_center=augument['featurewise_center'])
            testdatagen.fit(norm_images, seed=params['seed'])
            testmaskdatagen.fit(norm_masks,seed=params['seed'])

            self.image_iterator = datagen.flow(train_img, batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
            self.mask_iterator = maskdatagen.flow(train_msk, batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
            self.train_iterator = zip(self.image_iterator, self.mask_iterator)

            self.test_image_iterator = datagen.flow(test_img, batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
            self.test_mask_iterator = maskdatagen.flow(test_msk, batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
            self.test_iterator = zip(self.test_image_iterator, self.test_mask_iterator)

           

            print('Data loaded.')
            print(f'Training samples: {len(train_img)}, mean: {np.mean(train_img)},\n \
                Validation samples: {len(test_img)}, mean: {np.mean(test_img)}')

    def build_model(self) -> str:
        self.model = UNet(FEATURE_CHANNELS, IMAGE_SIZE)

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

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1)


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
        mlflow.log_artifact(path)
        print('Model weights saved: ' + path)



if __name__ == '__main__':
    tr = Trainer()
    tr.quick_load_data()
    tr.build_model()
    tr.train()
    tr.save()

