import numpy as np
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import cv2
from sklearn.metrics import jaccard_score, confusion_matrix
import random
import mlflow
import sys, os, datetime, time
from typing import Dict
import json
from UNet import UNet
from preprocessing import prep_cv as prep
import utils

#PARAMETERS
params = {
    'shuffle': True,
    'seed': 1
}

#SEEDS
random.seed(params['seed'])
np.random.seed(params['seed'])
tf.random.set_seed(params['seed'])

#HYPERPARAMETERS
IMAGE_SIZE = 128
TRAIN_PATH = 'src/models/'
EPOCHS = 1
BATCH_SIZE = 8
DATASET_SIZE = 2694 #Number of datapoints 
FEATURE_CHANNELS = [16, 32,64,128,256] #Number of feature channels at each floor of the UNet structure
LOG_DIRECTORY = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


class Trainer:
    '''Tensorflow model trainer.'''

    def __init__(self, debug_mode = False) -> None:
        #Base properties
        self.model = None
        self.debug_mode = debug_mode
        self.input_shape = (1,)
        self.augument = {
            'rotation_range': 10,
            'zoom_range': 0.1,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'horizontal_flip':True,
            'vertical_flip': True,
        }
        self.augument_mask = {
            'rotation_range': 10,
            'zoom_range': 0.1,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'horizontal_flip':True,
            'vertical_flip': True,
        }
        self.preprocessing_parameters = {
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
       
        #Callbacks init
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIRECTORY, histogram_freq=1)    
        self.early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=14, restore_best_weights=True, verbose=1)
        
        self.callbacks = [
            self.early_stopping_callback,
            self.tensorboard_callback
        ]

        #Optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        #Metrics
        self.metrics = [
            'acc',
            tf.keras.metrics.Precision(),
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Recall(),
        ]

        #Mlflow start
        self.__init_log_parameters()
            
    def __init_log_parameters(self) -> None:
        '''Initialize mlflow, and add some log parameters'''
        mlflow.tensorflow.autolog()
        mlflow.log_param('FEATURE_CHANNELS', FEATURE_CHANNELS)
        mlflow.log_param('IMAGE_SIZE', IMAGE_SIZE)
       
        if self.debug_mode == False:
            mlflow.log_params(params)
            mlflow.log_params(self.augument)
            mlflow.log_params(self.preprocessing_parameters)

    def __load_data_into_dict(self) -> None:
        '''Fetch data from numpy tables and create dictionary with train, val, test data'''
        dir = 'npy_datasets/cv_data128/'
        self.data = {}
        self.data['train_images'] =  np.load(dir + 'cv_train_images.npy')
        self.data['train_masks'] = np.load(dir + 'cv_train_masks.npy')
        self.data['val_images'] =  np.load(dir + 'cv_val_images.npy')
        self.data['val_masks'] =  np.load(dir + 'cv_val_masks.npy')
        self.data['test_images'] =  np.load(dir + 'cv_test_images.npy')
        self.data['test_masks'] =  np.load(dir + 'cv_test_masks.npy')

        print('Training set shape: ', self.data['train_images'].shape)

    def __apply_preprocessings(self) -> None:
        '''Apply preprocessings based on parameters contained in preprocessing_parameters dictionary'''
        #APPLY GAUSSIAN BLUR - performed on train/val/test input images
        if self.preprocessing_parameters['gaussian_blur']:
            for key in self.data.keys():
                if 'images' in key:
                    self.data[key] = utils.apply_gaussian_blur(self.data[key], self.preprocessing_parameters['gaussian_blur_radius'])
            print('Applied gaussian blur on all input images')

        #APPLY HISTOGRAM EQUALIZATION - performed on train/val/test input images
        if self.preprocessing_parameters['histogram_equalization']:
            for key in self.data.keys():
                if 'images' in key:
                    self.data[key] = utils.apply_histogram_equalization(self.data[key], self.preprocessing_parameters['histogram_cutoff_percentage'])
            print('Applied histogram equalization on all input images')

        #GET CONNECTED COMPONENTS
        if self.preprocessing_parameters['connected_components']:
            for key in self.data.keys():
                if 'images' in key:
                    self.data[key] = prep.connected_components_on_batch(np.array(self.data[key], dtype='f'), take=5)
            self.input_shape = self.data['train_images'].shape
            print('Applied and added connected components channels')

        #APPLY NORMALIZATION PER-CHANNEL
        if self.preprocessing_parameters['per_channel_normalization']:      
            self.data['train_images'] , mean = utils.norm_per_channel(self.data['train_images'] )
            self.data['val_images'], mean = utils.norm_per_channel(self.data['val_images'], mean)
            self.data['test_images'], mean = utils.norm_per_channel(self.data['test_images'], mean)
            mlflow.log_param('mean_per_channel', mean)
            print('Normalized per channel')

        #APPLY NORMALIZATION
        if self.preprocessing_parameters['normalization']:
            self.data['train_images'], self.data['train_masks'] = utils.normalize(self.data['train_images'], self.data['train_masks'])
            self.data['test_images'], self.data['test_masks'] = utils.normalize(self.data['test_images'], self.data['test_masks'])
            self.data['val_images'], self.data['val_masks'] = utils.normalize(self.data['val_images'], self.data['val_masks'])
            print('Applied normalization')

        #APPLY ZCA
        if self.preprocessing_parameters['zca_whitening']:
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

            #Fetch data from file and create dictionary
            self.__load_data_into_dict()            
            print('Data loaded')

            #Apply all preprocessings
            self.__apply_preprocessings()

            #Disable data augumentation
            if not self.preprocessing_parameters['augumentation']:
                self.augument = {}
                self.augument_mask ={}
                print('Data augumentation off')
            else:
                print('Data augumentation on')

            #Initialize generators
            datagen = ImageDataGenerator(**self.augument)
            datagen.fit(self.data['train_images'], seed=params['seed'])

            maskdatagen = ImageDataGenerator(**self.augument_mask)
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
        '''Build and compile tf model structure from custom UNet architecture.\n
        Return: model summary'''
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
        '''Save model weights into h5 format. \n
        Return: path to model.''' 
        timestamp = time.strftime(r"%d%m%Y-%H%M%S")
        path = f'{TRAIN_PATH}UNet_model_{IMAGE_SIZE}x{IMAGE_SIZE}_{timestamp}.h5'

        try:
            self.model.save_weights(path)
        except:
            print('Saved models directiory created')
            os.mkdir(f'./{TRAIN_PATH}')

        mlflow.log_param('Saved Model Name', path)
        print('Model weights saved: ' + path)
        return path

    def evaluate(self) -> None:  
        '''Evaluate and save model performance with metrics'''
        res = self.model.evaluate(
            x=self.data['test_images'], 
            y=self.data['test_masks'], 
            verbose=1, 
            batch_size=BATCH_SIZE
            )

        mlflow.log_param('Test performance',list(zip(self.model.metrics_names,res )))

    def test_model_additional_metrics(self) -> None:
        '''Test model for additional metrics: 
        sensitivity, specifitivity, jaccard index, isic score, dice.'''
        results = self.model.predict(self.data['test_images'])

        jacc_sum = tn = tp = fp = fn = test_jacc_sum = test_jacc_above_thresh= 0

        for r,mask in zip(results, self.data['test_masks']):
            
            d = mask > 0.5
            _, r = cv2.threshold( np.array(r*255, dtype='uint8'), 0, 255, cv2.THRESH_OTSU)
            r = (r / 255) > 0.5

            r = r.flatten()
            d = d.flatten()

            inter = np.logical_and(d,r).sum()
            union = np.logical_or(d,r).sum()

            jacc_sum += inter / union 

            #Additional metrics
            tn_, fp_, fn_, tp_ = confusion_matrix(d, r).ravel()
            tn += tn_ 
            tp += tp_
            fp += fp_
            fn += fn_

            test_jaccard_score = jaccard_score(d, r)
            if test_jacc_sum >= 0.65:
                test_jacc_above_thresh+=1

            test_jacc_sum += test_jaccard_score

        test_sensitivity = tp / (tp + fn) 
        test_specifitivity = tn / (tn + fp) 
        test_accuracy = (tp + tn) / (tp + tn + fp + fn) 
        test_dsc = 2*tp / (2*tp + fp + fn) 
        test_jaccard_score = test_jacc_sum / len(results)


        mean_jaccard_index = jacc_sum / len(results)
        print('---------------TEST METRICS----------------------')
        print('jaccard_index', mean_jaccard_index)
        print('test_sensitivity', test_sensitivity)
        print('test_specifitivity', test_specifitivity)
        print('test_accuracy', test_accuracy)
        print('test_jaccard_score', test_jaccard_score)
        print('test_dicecoef', test_dsc)
        print('isic_eval_score', test_jacc_above_thresh / len(results))
        print('---------------TEST METRICS----------------------')

        mlflow.log_metric('jaccard_index', mean_jaccard_index)
        mlflow.log_metric('test_sensitivity', test_sensitivity)
        mlflow.log_metric('test_specifitivity', test_specifitivity)
        mlflow.log_metric('test_accuracy', test_accuracy)
        mlflow.log_metric('test_jaccard_score', test_jaccard_score)
        mlflow.log_metric('test_dicecoef', test_dsc)
        mlflow.log_metric('isic_eval_score', test_jacc_above_thresh / len(results))

    def load_prep_settings_from_string(self, settings):
        '''Load preprocessing parameters from config string. 
        String should contain keywords such as BS, AUG, GAUS etc.'''

        if 'AUG' in settings:
            self.preprocessing_parameters['augumentation'] = True 
        if 'HEQ' in settings: 
            self.preprocessing_parameters['histogram_equalization'] = True 
        if 'PCN' in settings: 
            self.preprocessing_parameters['per_channel_normalization'] = True 
        if 'CC' in settings: 
            self.preprocessing_parameters['connected_components'] = True 
        if 'ZCA' in settings: 
            self.preprocessing_parameters['zca_whitening'] = True 
        if 'GAUS' in settings: 
            self.preprocessing_parameters['gaussian_blur'] = True 
        

def run_training_from_config():
    file = open('runs_config.json', 'r') 
    d = json.load(file)

    for i, run in enumerate(d['runs']):
        
        print('-------------------------------------------------------------')
        print(f'Run number {i}, name: {run}')
        print('-------------------------------------------------------------')

        tr = Trainer()
        tr.load_prep_settings_from_string(run)
        tr.load_data()
        tr.build_model()
        tr.train()
        tr.evaluate()
        tr.test_model_additional_metrics()
        tr.save()

        mlflow.end_run()
    
    print('All runs from config have been executed.')

if __name__ == '__main__':
    run_training_from_config()
    exit()

