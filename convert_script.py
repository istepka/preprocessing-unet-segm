import numpy as np
from PIL import Image
import os
from data_loader import DataLoader


source_path_img = './src/data/raw_img/images/'
filenames = os.listdir(source_path_img)

RESOLUTION = 256


dl = DataLoader()
v_images, v_masks = dl.get_dataset(resolution=RESOLUTION, 
                                source_path_img='./src/data/validation/images/images/',
                                source_path_mask='./src/data/validation/masks/masks/'
                                    )

images, masks = dl.get_dataset(resolution=RESOLUTION, 
                                source_path_img='./src/data/raw_img/images/',
                                source_path_mask='./src/data/raw_masks/masks/'
                                    )
mat = np.random.choice(a=[False, True], size=(len(images) + len(v_images)), p=[0.9, 0.1])


imgs = np.concatenate( (images, v_images), axis=0)
msks = np.concatenate((masks, v_masks), axis=0) 


train_imgs = imgs[~mat]
train_masks = msks[~mat]

test_imgs = imgs[mat]
test_masks = msks[mat]



train_data = np.array((train_imgs, train_masks))
test_data = np.array((test_imgs, test_masks))

train_len = len(train_data)
test_len = len(test_data)

print(f'Train {train_len}, Test {test_len}')

np.save(f'train_data_{train_len}_{RESOLUTION}', train_data)
np.save(f'test_data_{test_len}_{RESOLUTION}', test_data)
