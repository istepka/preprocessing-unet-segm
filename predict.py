from array import ArrayType
from train import Trainer
import matplotlib.pyplot as plt
import numpy as np
import utils

def predict(
    image,
    model_path = 'src/models/UNet_model_256x256_22092021-130803.h5',
    preprocessings = {
        'histogram_equalization': True,
        'augumentation': True
    }
    ) -> ArrayType:
    '''Predict on the given image (or batch of images). \n
    Input image should have one grayscale channel with values 0-255.\n
    Return: `predicted_image(s)` without threshold applied'''

    if len(image.shape) < 4:
        image = np.reshape( image, (-1, *(image.shape)))
        print('Reshaped image ', image.shape)

    print(image.shape, np.max(image))

    # #Apply preprocessings based on parameters
    if 'histogram_equalization' in preprocessings.keys():
        image = utils.apply_histogram_equalization(image, 2)
    # # if 'per_channel_normalization' in preprocessings.keys():
    # #     image = utils.norm_per_channel(image)
    
    image = image / 255
    print(image.shape)

    trainer = Trainer(preprocessing_params=preprocessings) 
    trainer.build_model()
    trainer.model.load_weights(model_path)
    result = trainer.model.predict(image)

    print('Predicted.')
    return result

if __name__ == '__main__':
    images = np.load('npy_datasets/new_lesion/cv_test_images.npy')
    predicted = predict(images[0])
    utils.display([predicted*255, images[0]])
    
    
