import numpy as np
from PIL import Image
import os
from data_loader import DataLoader


source_path_img = './src/data/raw_img/images/'
filenames = os.listdir(source_path_img)

# RESOLUTION = 64


# dl = DataLoader()
# v_images, v_masks = dl.get_dataset(resolution=RESOLUTION, 
#                                 source_path_img='./src/data/validation/images/images/',
#                                 source_path_mask='./src/data/validation/masks/masks/'
#                                     )
# np.save('zca_test_64' ,np.array( (v_images, v_masks)) )
# images, masks = dl.get_dataset(resolution=RESOLUTION, 
#                                 source_path_img='./src/data/raw_img/images/',
#                                 source_path_mask='./src/data/raw_masks/masks/'
#                                     )
# mat = np.random.choice(a=[False, True], size=(len(images) + len(v_images)), p=[0.9, 0.1])


# imgs = np.concatenate( (images, v_images), axis=0)
# msks = np.concatenate((masks, v_masks), axis=0) 


# train_imgs = imgs[~mat]
# train_masks = msks[~mat]

# test_imgs = imgs[mat]
# test_masks = msks[mat]



# train_data = np.array((train_imgs, train_masks))
# test_data = np.array((test_imgs, test_masks))

# train_len = len(train_data)
# test_len = len(test_data)

# print(f'Train {train_len}, Test {test_len}')

# np.save(f'train_data_{train_len}_{RESOLUTION}', train_data)
# np.save(f'test_data_{test_len}_{RESOLUTION}', test_data)

import cv2 as cv
import preprocessing.prep_cv as prep
RES=128
source_path_img = './src/data/raw_img/images/'
source_path_mask = './src/data/raw_masks/masks/'
source_path_val_img = './src/data/validation/images/images/'
source_path_val_mask = './src/data/validation/masks/masks/'
#filenames = os.listdir(source_path_img)
filenames = list(map(lambda x: x[:-4] ,os.listdir(source_path_img)))

print(f'found {len(filenames)} images')

images = list()
masks = list()

for i,name in enumerate(filenames):

    im  = cv.imread(source_path_img + name + '.jpg')
    #print(source_path_val_img + name + '.jpg')
    c = cv.resize(im, (RES,RES))
    c = cv.cvtColor(c, cv.COLOR_RGB2GRAY)

    #cc = prep.connected_components(c, take=3, debug=False)

    c = np.reshape(c, (RES,RES,-1))

    # for ch in cc:
    #     ch = np.reshape(ch, (256,256,-1))
    #     c = np.concatenate((c, ch), axis=2)


    mask = cv.imread(source_path_mask + name + '_segmentation.png')
    mask = cv.resize(mask, (RES,RES))
    mask = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
    mask = np.reshape(mask, (RES,RES,-1))

    images.append(c)
    masks.append(mask)

    if i%50==0:
        print(i)
    



images = np.array(images)
masks = np.array(masks)

print(images.shape)
print(masks.shape)


np.save('npy_datasets/cv_data128/cv_images.npy', images)

np.save('npy_datasets/cv_data128/cv_masks.npy', masks)


