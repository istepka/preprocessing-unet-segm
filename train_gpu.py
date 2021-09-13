import numpy as np
import tensorflow as tf 
from data_loader import DataLoader
import random
import utils
import time
import mlflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from UNet import UNet
import datetime
import sys


#SEEDS
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

#HYPERPARAMETERS
IMAGE_SIZE = 128
TRAIN_PATH = 'src/models/'
EPOCHS = 2
BATCH_SIZE = 8
DATASET_SIZE = 2694 #Number of datapoints 
FEATURE_CHANNELS = [32,64,128,256,512] #Number of feature channels at each floor of the UNet structure
LOG_DIRECTORY = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#PARAMETERS
params = {
    'shuffle': True,
    'seed': 1
}
augument = {
    'rotation_range': 10,
    'zoom_range': 0.1,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip':True,
    'vertical_flip': True,
}
augument_mask = {
    'rotation_range': 10,
    'zoom_range': 0.1,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip':True,
    'vertical_flip': True,
}
preprocessing_parameters = {
    'augumentation': True,
    'normalization': True,
    'histogram_equalization': False,
    'histogram_cutoff_percentage': 0.02,
    'remove_vignette_algorithm': False,
    'add_border': False,
    'border_width': 20,
    'per_channel_normalization': False,
    'gaussian_blur': False,
    'gaussian_blur_radius': 2,
    'zca_whitening': False
}


class Trainer:
    '''Tensorflow model trainer'''

    def __init__(self) -> None:
        #Base properties
        self.model = None
        self.__init_log_parameters()
       
        #Callbacks init
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIRECTORY, histogram_freq=1)    
        self.early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)

        #Optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        #Metrics
        self.metrics = [
            'acc',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Recall(),
            #utils.mean_iou
        ]

        #Callbacks
        self.callbacks = [
            self.early_stopping_callback,
            self.tensorboard_callback
        ]
        
    def __init_log_parameters(self) -> None:
        mlflow.tensorflow.autolog()
        mlflow.log_param('FEATURE_CHANNELS', FEATURE_CHANNELS)
        mlflow.log_param('IMAGE_SIZE', IMAGE_SIZE)
        mlflow.log_param('BATCH_SIZE', BATCH_SIZE)
        mlflow.log_param('DATA_AUGUMENTATION', preprocessing_parameters['augumentation'])
        mlflow.log_params(params)
        mlflow.log_params(augument)
        mlflow.log_params(preprocessing_parameters)

    def load_data(self) -> None:
        '''Load data from numpy table file, create generators and apply preprocessing according to parameters.'''

        with tf.device('/device:GPU:0'):

            train_img, train_msk = DataLoader().get_data_from_npy('128new_train_data_2242.npy')
            test_imgs, test_msks = DataLoader().get_data_from_npy('128new_test_data_246.npy')
            val_imgs, val_msks = DataLoader().get_data_from_npy('128new_val_data_206.npy')
            print('Data loaded')

            #APPLY GAUSSIAN BLUR
            if preprocessing_parameters['gaussian_blur']:
                train_img = utils.apply_gaussian_blur(train_img, preprocessing_parameters['gaussian_blur_radius'])
                test_imgs = utils.apply_gaussian_blur(test_imgs, preprocessing_parameters['gaussian_blur_radius'])
                val_imgs = utils.apply_gaussian_blur(val_imgs, preprocessing_parameters['gaussian_blur_radius'])
                print('Applied gaussian blur')

            #APPLY HISTOGRAM EQUALIZATION
            if preprocessing_parameters['histogram_equalization']:
                train_img = utils.apply_histogram_equalization(train_img, preprocessing_parameters['histogram_cutoff_percentage'])
                test_imgs = utils.apply_histogram_equalization(test_imgs, preprocessing_parameters['histogram_cutoff_percentage'])
                val_imgs = utils.apply_histogram_equalization(val_imgs, preprocessing_parameters['histogram_cutoff_percentage'])
                print('Applied histogram equalization')

            #APPLY NORMALIZATION PER-CHANNEL
            if preprocessing_parameters['per_channel_normalization']:
                train_img, mean = utils.norm_per_channel(train_img)
                test_imgs, mean = utils.norm_per_channel(test_imgs, mean)
                val_imgs, mean = utils.norm_per_channel(val_imgs, mean)
                mlflow.log_param('mean_per_channel', mean)
                print('Normalized per channel')

            #APPLY NORMALIZATION
            if preprocessing_parameters['normalization']:
                train_img, train_msk = utils.normalize(train_img, train_msk)
                test_imgs, test_msks = utils.normalize(test_imgs, test_msks)
                val_imgs, val_msks = utils.normalize(val_imgs, val_msks)
                print('Applied normalization')

            if preprocessing_parameters['zca_whitening']:
                gen = ImageDataGenerator(featurewise_center=True ,zca_whitening=True)
                print('ZCA fit will be performed, it might take some time')
                gen.fit(val_imgs[0:100], seed=133)
                print('ZCA fit done')
               

                for i in range(len(train_img)):
                    train_img[i] = gen.standardize(train_img[i])

                    if i < len(val_imgs):
                        val_imgs[i] = gen.standardize(val_imgs[i])
                    if i < len(test_imgs):
                        test_imgs[i] = gen.standardize(test_imgs[i])

                    if i % 50 == 0:
                        print('ZCA applied to ', i, ' samples')

                # train_img = utils.apply_zca_normalization(train_img)
                # val_imgs = utils.apply_zca_normalization(val_imgs)
                # test_imgs = utils.apply_zca_normalization(test_imgs)
                print('Applied zca_whitening')


            #APPLY VALIDATION SPLIT
            #train_img, train_msk, test_img, test_msk = utils.split_train_test(norm_images, norm_masks, validation_split=0.85)
            #print(f'Train data {len(train_img)}\nValidation data {len(test_img)}')

            self.test_images = test_imgs
            self.test_masks = test_msks

            #APPLY AUGUMENTATION
            if preprocessing_parameters['augumentation']:
                print('Data augumentation on')

                #Initialize generators
                datagen = ImageDataGenerator(**augument)
                datagen.fit(train_img, seed=params['seed'])

                maskdatagen = ImageDataGenerator(**augument_mask)
                maskdatagen.fit(train_msk,seed=params['seed'])

                validdatagen = ImageDataGenerator()

                #Initialize flow and zip it to format proper to fit() function 
                image_iterator = datagen.flow(train_img, batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
                mask_iterator = maskdatagen.flow(train_msk, batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
                self.train_iterator = zip(image_iterator, mask_iterator)
                self.train_steps = len(image_iterator)

                valid_image_iterator = validdatagen.flow(val_imgs, batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
                valid_mask_iterator = validdatagen.flow(val_msks, batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
                self.valid_iterator = zip(valid_image_iterator, valid_mask_iterator)
                self.valid_steps = len(valid_image_iterator)

            

            print('Prep done')
            print(f'Training samples: {len(train_img)}, channel mean: {np.mean(train_img)},\nValidation samples: {len(val_imgs)}, channel mean: {np.mean(val_imgs)}')

    def build_model(self) -> str:
        '''Build and compile tf model structure from custom UNet architecture.'''
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
        '''Train model with fit() function.'''
        with tf.device('/device:GPU:0'):
            self.model.fit(
                self.train_iterator,
                steps_per_epoch=self.train_steps ,
                epochs=EPOCHS, 
                validation_steps=self.valid_steps,
                validation_data=self.valid_iterator,
                callbacks=self.callbacks
            )

    def save(self) -> str:   
        '''Save model weights into h5 format''' 
        timestamp = time.strftime(r"%d%m%Y-%H%M%S")
        path = f'{TRAIN_PATH}UNet_model_{IMAGE_SIZE}x{IMAGE_SIZE}_{timestamp}.h5'
        self.model.save_weights(path)
        mlflow.log_param('Saved Model Name', path)
        print('Model weights saved: ' + path)
        return path

    def evaluate(self) -> None:  
        res = self.model.evaluate(x=self.test_images, y=self.test_masks, verbose=1, batch_size=BATCH_SIZE)
        mlflow.log_param('Test performance',list(zip(self.model.metrics_names,res )))

if __name__ == '__main__':
    tmp_augument = augument
    tmp_augument_mask = augument_mask

    for i in range(0,1):
        run_name = 'BS+A'
        preprocessing_parameters['gaussian_blur'] = False 
        preprocessing_parameters['histogram_equalization'] = False  
        preprocessing_parameters['per_channel_normalization'] = False 
        #augument = {} 
        #augument_mask = {}

        preprocessing_parameters['zca_whitening'] = False

        # if i == 0:  
        #     preprocessing_parameters['gaussian_blur'] = False 
        #     preprocessing_parameters['histogram_equalization'] = False  
        #     preprocessing_parameters['per_channel_normalization'] = False 
        #     augument = {}
        #     augument_mask = {}
        #     run_name = 'V'
        # if i == 1:
        #     preprocessing_parameters['per_channel_normalization'] = True
        #     augument = {}
        #     augument_mask = {}
        #     run_name = 'V+PCN'
        # if i == 2:
        #     augument = tmp_augument
        #     augument_mask = tmp_augument_mask
        #     run_name = 'V+PCN+A'
        # if i == 3:
        #     preprocessing_parameters['histogram_equalization'] = True  
        #     run_name = 'V+PCN+A+HEQ'
        # if i == 4:
        #     preprocessing_parameters['gaussian_blur'] = True  
        #     run_name = 'V+PCN+A+HEQ+GB'


        mlflow.start_run(run_name=run_name)
        if len(sys.argv) > 1:
            mlflow.log_param('Run command line description', str(sys.argv[1:]))

        tr = Trainer()
        tr.load_data()
        tr.build_model()
        tr.train()
        tr.evaluate()

        #mlflow.keras.save_model(tr.model, f'C:/Users/ignac/OneDrive/Pulpit/Lesion-boundary-segmentation/mlruns/model{time.strftime(r"%d%m%Y-%H%M%S")}')

        tr.save()

        mlflow.end_run()

