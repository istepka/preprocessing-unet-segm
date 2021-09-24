import numpy as np
from PIL import Image
import os
import cv2 as cv

RES=256
source_path_img = './src/data/2017/data/'
source_path_mask = './src/data/2017/truth/'
source_path_val_img = './src/data/validation/images/images/'
source_path_val_mask = './src/data/validation/masks/masks/'

filenames = list(map(lambda x: x[:-4] ,os.listdir(source_path_img)))

print(f'found {len(filenames)} images')

images = list()
masks = list()

for i,name in enumerate(filenames):

    if 'super' in name:
        continue
    #print(source_path_img + name + '.jpg')
    im  = cv.imread(source_path_img + name + '.jpg')
   
    c = cv.resize(im, (RES,RES))
    c = cv.cvtColor(c, cv.COLOR_RGB2GRAY)

    c = np.reshape(c, (RES,RES,-1))

    # for ch in cc:
    #     ch = np.reshape(ch, (256,256,-1))
    #     c = np.concatenate((c, ch), axis=2)


    mask = cv.imread(source_path_mask + name + '_segmentation.png')
    mask = cv.resize(mask, (RES,RES)) * 255
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


np.save('npy_datasets/2017/cv_images.npy', images)

np.save('npy_datasets/2017/cv_masks.npy', masks)


