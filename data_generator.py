from utils import split_train_test
from PIL import Image
import numpy as np
import os
import tensorflow as tf



class DataGenerator(tf.keras.utils.Sequence):
    '''Generates data for Keras model'''

    def __init__(self, mode, batch_size=16, img_dim=(512,512), n_channels=1, shuffle=True) -> None:
        '''Initialization'''
        self.img_dim = img_dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.filepath = 'src/data/'
        self.__get_ID_list(mode)
        self.on_epoch_end()

    def __get_ID_list(self, mode):
        '''Load the list of image IDs avaliable in data directory '''
        self.list_IDs = list(map(lambda x: x[:-4] ,os.listdir(self.filepath + 'raw_img')))
        train_split_percentage = 0.2
        datapoints = len(self.list_IDs)
        split_index = int(datapoints * train_split_percentage)

        #Depending on mode reuturn right ids
        if mode == 'train':
            self.list_IDs = self.list_IDs[0:split_index]
        elif mode == 'validation':
            self.list_IDs = self.list_IDs[split_index: ]
        else:
            pass
        
        
           

    def on_epoch_end(self):
        '''Shuffles and updates indexes after each epoch'''
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        '''Generates data containing batch-size samples'''
        #Initialize arrays
        X = np.empty( (self.batch_size, *self.img_dim, self.n_channels) )
        Y = np.empty( (self.batch_size, *self.img_dim, self.n_channels) )

        #Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            x = Image.open(f'{self.filepath}raw_img/{ID}.jpg').convert(mode='L').resize((512,512))
            y = Image.open(f'{self.filepath}raw_masks/{ID}_segmentation.png').convert(mode='L').resize((512,512))

            x = np.asarray(x.getdata()).reshape(512,512, -1) 
            y = np.asarray(y.getdata()).reshape(512,512, -1)

            x = x / 255
            y = (y > 0).astype(float)

            X[i,] = x
            Y[i,] =  y

        return X, Y

    def __len__(self):
        '''Gives the number of batches per epoch'''
        return int(np.floor( len(self.list_IDs) / self.batch_size  ))


    def __getitem__(self, index):
        '''Generate one batch of the data'''
        #Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]

        #Get the list of ids
        list_IDs_temp = [ self.list_IDs[i] for i in indexes ]

        #Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y
            