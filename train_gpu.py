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
from preprocessing import prep_cv as prep


#SEEDS
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

#HYPERPARAMETERS
IMAGE_SIZE = 256
TRAIN_PATH = 'src/models/'
EPOCHS = 100
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
    'augumentation': False,
    'normalization': True,
    'histogram_equalization': False,
    'histogram_cutoff_percentage': 0.02,
    'remove_vignette_algorithm': False,
    'add_border': False,
    'border_width': 20,
    'per_channel_normalization': False,
    'gaussian_blur': False,
    'gaussian_blur_radius': 2,
    'zca_whitening': False,
    'connected_components': False
} #By default all preprocessings are disabled except 0-1 normalization


class Trainer:
    '''Tensorflow model trainer'''

    def __init__(self) -> None:
        #Base properties
        self.model = None
        self.input_shape = 1
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
        '''Initialize mlflow, and add some log parameters'''
        mlflow.tensorflow.autolog()
        mlflow.log_param('FEATURE_CHANNELS', FEATURE_CHANNELS)
        mlflow.log_param('IMAGE_SIZE', IMAGE_SIZE)
        mlflow.log_param('BATCH_SIZE', BATCH_SIZE)
        mlflow.log_param('DATA_AUGUMENTATION', preprocessing_parameters['augumentation'])
        mlflow.log_params(params)
        mlflow.log_params(augument)
        mlflow.log_params(preprocessing_parameters)

    def __load_data_into_dict(self) -> None:
        '''Fetch data from numpy tables and create dictionary with train, val, test data'''
        self.data = {}
        self.data['train_images'] =  np.load('npy_datasets/cv_data/cv_train_images.npy')
        self.data['train_masks'] = np.load('npy_datasets/cv_data/cv_train_masks.npy')
        self.data['val_images'] =  np.load('npy_datasets/cv_data/cv_val_images.npy')
        self.data['val_masks'] =  np.load('npy_datasets/cv_data/cv_val_masks.npy')
        self.data['test_images'] =  np.load('npy_datasets/cv_data/cv_test_images.npy')
        self.data['test_masks'] =  np.load('npy_datasets/cv_data/cv_test_masks.npy')

        print(self.data['train_images'].shape)

    def __apply_preprocessings(self) -> None:
        '''Apply preprocessings based on parameters contained in preprocessing_parameters dictionary'''
        #APPLY GAUSSIAN BLUR - performed on train/val/test input images
        if preprocessing_parameters['gaussian_blur']:
            for key in self.data.keys():
                if 'images' in key:
                    self.data[key] = utils.apply_gaussian_blur(self.data[key], preprocessing_parameters['gaussian_blur_radius'])
            print('Applied gaussian blur on all input images')

        #APPLY HISTOGRAM EQUALIZATION - performed on train/val/test input images
        if preprocessing_parameters['histogram_equalization']:
            for key in self.data.keys():
                if 'images' in key:
                    self.data[key] = utils.apply_histogram_equalization(self.data[key], preprocessing_parameters['histogram_cutoff_percentage'])
            print('Applied histogram equalization on all input images')

        #GET CONNECTED COMPONENTS
        if preprocessing_parameters['connected_components']:
            for key in self.data.keys():
                if 'images' in key:
                    self.data[key] = prep.connected_components_on_batch(np.array(self.data[key], dtype='f'), take=5)
            self.input_shape = self.data['train_images'].shape
            print('Applied and added connected components channels')

        #APPLY NORMALIZATION PER-CHANNEL
        if preprocessing_parameters['per_channel_normalization']:      
            self.data['train_images'] , mean = utils.norm_per_channel(self.data['train_images'] )
            self.data['val_images'], mean = utils.norm_per_channel(self.data['val_images'], mean)
            self.data['test_images'], mean = utils.norm_per_channel(self.data['test_images'], mean)
            mlflow.log_param('mean_per_channel', mean)
            print('Normalized per channel')

        #APPLY NORMALIZATION
        if preprocessing_parameters['normalization']:
            self.data['train_images'], self.data['train_masks'] = utils.normalize(self.data['train_images'], self.data['train_masks'])
            self.data['test_images'], self.data['test_masks'] = utils.normalize(self.data['test_images'], self.data['test_masks'])
            self.data['val_images'], self.data['val_masks'] = utils.normalize(self.data['val_images'], self.data['val_masks'])
            print('Applied normalization')

        #APPLY ZCA
        if preprocessing_parameters['zca_whitening']:
            gen = ImageDataGenerator(featurewise_center=True ,zca_whitening=True)
            print('ZCA fit will be performed, it might take some time')
            gen.fit(self.data['train_images'][0:250], seed=133)
            print('ZCA fit done')
            

            for i in range(len(self.data['train_images'])):
                self.data['train_images'][i] = gen.standardize(self.data['train_images'][i])

                if i < len(self.data['val_images']):
                    self.data['val_images'][i] = gen.standardize(self.data['val_images'][i])
                if i < len(self.data['test_images']):
                    self.data['test_images'][i] = gen.standardize(self.data['test_images'][i])

                if i % 50 == 0:
                    print('ZCA applied to ', i, ' samples')

            print('Applied zca_whitening')

    def load_data(self) -> None:
        '''Load data, create generators and apply preprocessing according to parameters.'''

        with tf.device('/device:GPU:0'):

            #Fetch data from file and create dictionary from it
            self.__load_data_into_dict()            
            print('Data loaded')

            #Apply all preprocessings
            self.__apply_preprocessings()

            #Disable data augumentation
            if not preprocessing_parameters['augumentation']:
                augument = {}
                augument_mask ={}
                print('Data augumentation off')
            else:
                print('Data augumentation on')

            #Initialize generators
            datagen = ImageDataGenerator(**augument)
            datagen.fit(self.data['train_images'], seed=params['seed'])

            maskdatagen = ImageDataGenerator(**augument_mask)
            maskdatagen.fit(self.data['train_masks'],seed=params['seed'])

            validdatagen = ImageDataGenerator()

            #Initialize flow and zip it to format proper to fit() function 
            image_iterator = datagen.flow(self.data['train_images'], batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
            mask_iterator = maskdatagen.flow(self.data['train_masks'], batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
            self.train_iterator = zip(image_iterator, mask_iterator)
            self.train_steps = len(image_iterator)

            valid_image_iterator = validdatagen.flow(self.data['val_images'], batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
            valid_mask_iterator = validdatagen.flow(self.data['val_masks'], batch_size=BATCH_SIZE, shuffle=params['shuffle'], seed=params['seed'])
            self.valid_iterator = zip(valid_image_iterator, valid_mask_iterator)
            self.valid_steps = len(valid_image_iterator)

            print('Prep done')
            print(f'Training samples: {len(self.data["train_images"])}, channel mean: {np.mean(self.data["train_images"])},\nValidation samples: {len(self.data["val_images"])}, channel mean: {np.mean(self.data["val_images"])}')

    def build_model(self) -> str:
        '''Build and compile tf model structure from custom UNet architecture.'''
        with tf.device('/device:GPU:0'):
            self.model = UNet(FEATURE_CHANNELS, IMAGE_SIZE, self.input_shape[-1])  
            
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
        '''Evaluate and save model performance with metrics'''
        res = self.model.evaluate(x=self.data['test_images'], y=self.data['test_masks'], verbose=1, batch_size=BATCH_SIZE)
        mlflow.log_param('Test performance',list(zip(self.model.metrics_names,res )))

if __name__ == '__main__':


    for i in range(0,1):
        run_name = 'BS+HEQ+CC(5 additional channels)'

        preprocessing_parameters['connected_components'] = True
        preprocessing_parameters['histogram_equalization'] = True
        preprocessing_parameters['normalization'] = True

     

        mlflow.start_run(run_name=run_name)
        if len(sys.argv) > 1:
            mlflow.log_param('Run command line description', str(sys.argv[1:]))

        tr = Trainer()
        tr.load_data()
        tr.build_model()
        tr.train()
        tr.evaluate()

        tr.save()

        mlflow.end_run()

