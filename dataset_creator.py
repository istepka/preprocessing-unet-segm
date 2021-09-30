import numpy as np
import os
import cv2 

def create_dataset(
    source_path_img = './src/data/raw_img/images/',
    source_path_mask = './src/data/raw_masks/masks/',
    save_location = './npy_datasets/data/',
    resolution = 256,
    train_val_test_split = (0.7,0.15,0.15)
    ) -> str:
    '''Create dataset in form of numpy table files (.npy).\n
    Return: `save_location`'''

    filenames = list(map(lambda x: x[:-4] ,os.listdir(source_path_img)))
    print(f'found {len(filenames)} images')

    images = list()
    masks = list()

    for i,name in enumerate(filenames):
        im  = cv2.imread(source_path_img + name + '.jpg') 
        c = cv2.resize(im, (resolution,resolution))
        c = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
        c = np.reshape(c, (resolution,resolution,-1))


        mask = cv2.imread(source_path_mask + name + '_segmentation.png')
        mask = cv2.resize(mask, (resolution,resolution)) * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask = np.reshape(mask, (resolution,resolution,-1))

        images.append(c)
        masks.append(mask)

        if i%50==0:
            print(i)
        
    images = np.array(images)
    masks = np.array(masks)
    print(images.shape)
    print(masks.shape)

    #Shuffled
    p = np.random.permutation(len(images))
    images = images[p]
    masks = masks[p]

    mat = np.random.choice(a=[0, 1, 2], size=(len(images)), p=[*train_val_test_split])

    train_images = images[mat == 0]
    train_masks = masks[mat == 0]

    validation_images = images[mat == 1]
    validation_masks = masks[mat == 1]

    test_images = images[mat == 2]
    test_masks = masks[mat == 2]


    print(f'Train {len(train_images)}, \
        Validation {len(validation_images)}, \
        Test {len(test_images)}')

    if not os.path.exists(save_location):
        os.mkdir(save_location)

    np.save(save_location + 'cv_train_images.npy', train_images)
    np.save(save_location + 'cv_train_masks.npy', train_masks)
    np.save(save_location + 'cv_val_images.npy', validation_images)
    np.save(save_location + 'cv_val_masks.npy', validation_masks)
    np.save(save_location + 'cv_test_images.npy', test_images)
    np.save(save_location + 'cv_test_masks.npy', test_masks)

    print('All the data has been saved to ' + save_location)
    return save_location