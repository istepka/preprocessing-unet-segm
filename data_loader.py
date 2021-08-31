from typing import Tuple
from PIL import Image
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 


class DataLoader:

    def __init__(self) -> None:
        self.images = list()
        self.masks = list()
        

    def get_dataset(self, resolution=512, n=None):
        source_path_img = './src/data/raw_img/images/'
        source_path_mask = './src/data/raw_masks/masks/'

        if n is not None:
            filenames = list(map(lambda x: x[:-4] ,os.listdir(source_path_img)))[0:n+1]
        else:
            filenames = list(map(lambda x: x[:-4] ,os.listdir(source_path_img)))
       

        for i, name in enumerate(filenames):
            image = Image.open(source_path_img + name + '.jpg').convert(mode='L').resize((resolution,resolution))
            mask = Image.open(source_path_mask + name + '_segmentation.png').convert(mode='L').resize((resolution,resolution))

            image = np.asarray(image.getdata()).reshape(image.size[1], image.size[0], -1) 
            mask = np.asarray(mask.getdata()).reshape(mask.size[1], mask.size[0], -1)

            #image, mask = self.__normalize(image, mask)

            self.images.append(image)
            self.masks.append(mask)

            if i % 25 - 1 == 0:
                print(f'Loaded {i} images.')

        
      
        return np.array(self.images), np.array(self.masks)
        

    def tf_get_generators(self, resolution=256, batch_size=16) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
        
        data_gen_args = dict(rescale=1./255) 
        seed=1

        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        valid_datagen = ImageDataGenerator(**data_gen_args)
        valid_mask_datagen = ImageDataGenerator(**data_gen_args)


        image_generator = image_datagen.flow_from_directory(
            './src/data/raw_img',
            target_size=(resolution, resolution),
            color_mode='grayscale',
            seed=seed,
            class_mode=None,
            batch_size=batch_size,
            shuffle=True
        )

        mask_generator = mask_datagen.flow_from_directory(
            'src/data/raw_masks',
            target_size=(resolution, resolution),
            color_mode='grayscale',
            seed=seed,
            class_mode=None,
            batch_size=batch_size,
            shuffle=True
        )

        valid_generator = valid_datagen.flow_from_directory(
            'src/data/validation/images',
            target_size=(resolution, resolution),
            color_mode='grayscale',
            seed=seed,
            class_mode=None,
            batch_size=batch_size,
            shuffle=True
        )

        valid_mask_generator = valid_mask_datagen.flow_from_directory(
            'src/data/validation/masks',
            target_size=(resolution, resolution),
            color_mode='grayscale',
            seed=seed,
            class_mode=None,
            batch_size=batch_size,
            shuffle=True
        )

        return image_generator, mask_generator, valid_generator, valid_mask_generator
        

    def get_data_from_npy(self, filename=None):
        if filename is None:
            return np.load('data.npy')
        else:
            return np.load(filename)



if __name__ == "__main__":
    x,y,z = DataLoader().get_data_from_npy()
    print(len(x))
  