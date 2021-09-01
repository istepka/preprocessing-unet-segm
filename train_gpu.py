from preprocessing.preprocessor import Preprocessor
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
from PIL import Image

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
FEATURE_CHANNELS = [32,64,128,256,512] #Number of feature channels at each floor of the UNet structure

#PARAMETERS
params = {
    'shuffle': True,
    'seed': 1
}
augument = {
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
preprocessing_parameters = {
    'augumentation': True,
    'normalization': True,
    'histogram_equalization': True,
    'histogram_cutoff_percentage': 0.02,
    'remove_vignette_algorithm': False,
    'add_border': False,
    'border_width': 20,
    'per_channel_normalization': True,
    'gaussian_blur': True,
    'gaussian_blur_radius': 2
}


class Trainer:
    '''Tensorflow model trainer'''

    def __init__(self) -> None:
        self.model = None
        self.__init_log_parameters()
        self.metrics = [
            'acc',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.AUC(),
            #utils.iou,
            #utils.jaccard_index,
            #tf.keras.metrics.MeanIoU(num_classes=2),
            #tf.keras.metrics.TrueNegatives(),
            #tf.keras.metrics.TruePositives()       
        ]
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
        ]
        self.optimizer = tf.keras.optimizers.Adam()

    def __init_log_parameters(self):
        mlflow.tensorflow.autolog()
        mlflow.log_param('FEATURE_CHANNELS', FEATURE_CHANNELS)
        mlflow.log_param('IMAGE_SIZE', IMAGE_SIZE)
        mlflow.log_param('DATA_AUGUMENTATION', preprocessing_parameters['augumentation'])
        mlflow.log_param('BATCH_SIZE', BATCH_SIZE)
        mlflow.log_params(params)
        mlflow.log_params(augument)
        mlflow.log_params(preprocessing_parameters)

    def load_data(self):
        '''Load data from numpy table file'''
        with tf.device('/device:GPU:0'):
            imgs, msks = DataLoader().get_data_from_npy('data.npy')
            val_imgs, val_msks = DataLoader().get_data_from_npy('data_validation.npy')

            imgs = np.concatenate( (imgs, val_imgs), axis=0)
            msks = np.concatenate((msks, val_msks), axis=0) 
            
            #APPLY GAUSSIAN BLUR
            if preprocessing_parameters['gaussian_blur']:
                imgs = utils.apply_gaussian_blur(imgs, preprocessing_parameters['gaussian_blur_radius'])
                print('Applied gaussian blur')

            #APPLY HISTOGRAM EQUALIZATION
            if preprocessing_parameters['histogram_equalization']:
                imgs = utils.apply_histogram_equalization(imgs, preprocessing_parameters['histogram_cutoff_percentage'])
                print('Applied histogram equalization')

            #APPLY NORMALIZATION PER-CHANNEL
            if preprocessing_parameters['per_channel_normalization']:
                imgs, mean = utils.norm_per_channel(imgs)
                mlflow.log_param('mean_per_channel', mean)
                print('Normalized per channel')

            #APPLY NORMALIZATION
            if preprocessing_parameters['normalization']:
                norm_images, norm_masks = utils.normalize(imgs, msks)
                print('Applied normalization')

            

            #APPLY VALIDATION SPLIT
            train_img, train_msk, test_img, test_msk = utils.split_train_test(norm_images, norm_masks, validation_split=0.85)
            
            
            print(f'Train data {len(train_img)}\nValidation data {len(test_img)}')

            #APPLY AUGUMENTATION
            if preprocessing_parameters['augumentation']:
                print('Data augumentation on')
                datagen = ImageDataGenerator(**augument)
                datagen.fit(train_img, seed=params['seed'])

                maskdatagen = ImageDataGenerator(**augument_mask)
                maskdatagen.fit(train_msk,seed=params['seed'])

                testdatagen = ImageDataGenerator()
                #testmaskdatagen = ImageDataGenerator(featurewise_center=augument['featurewise_center'])
                #testdatagen.fit(norm_images, seed=params['seed'])
                #testmaskdatagen.fit(norm_masks,seed=params['seed'])

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
        with tf.device('/device:GPU:0'):
            self.model = UNet(FEATURE_CHANNELS, IMAGE_SIZE)  
            
            self.model.compile(
                optimizer=self.optimizer, 
                loss='binary_crossentropy', 
                metrics=self.metrics
                )
            
            print('Model built.')
        return self.model.summary()

    def train(self) -> None:
        with tf.device('/device:GPU:0'):
            self.model.fit(
                self.train_iterator,
                steps_per_epoch=len(self.image_iterator),
                epochs=EPOCHS, 
                validation_steps=len(self.test_image_iterator),
                validation_data=self.test_iterator,
                callbacks=self.callbacks
            )

    def save(self) -> None:    
        timestamp = time.strftime(r"%d%m%Y-%H%M%S")
        path = f'{TRAIN_PATH}UNet_model_{IMAGE_SIZE}x{IMAGE_SIZE}_{timestamp}.h5'
        self.model.save_weights(path)
        mlflow.log_param('Saved Model Name', path)
        print('Model weights saved: ' + path)



if __name__ == '__main__':
    tr = Trainer()
    tr.load_data()
    tr.build_model()
    tr.train()
    tr.save()

