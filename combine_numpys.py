import numpy as np 


im = np.load('npy_datasets/cv_data/cv_images.npy')
im_val = np.load('npy_datasets/cv_data/cv_images_val.npy')

mask = np.load('npy_datasets/cv_data/cv_masks.npy')
mask_val = np.load('npy_datasets/cv_data/cv_masks_val.npy')

print('Loaded data')

#Combined
images = np.concatenate((im, im_val))
masks = np.concatenate((mask, mask_val))


p = np.random.permutation(len(images))

#Shuffled
images = images[p]
masks = masks[p]


#Extract train_test_val
mat = np.random.choice(a=[False, True], size=(len(images)), p=[0.85, 0.15])

validation_images = images[mat]
validation_masks = masks[mat]

images = images[~mat]
masks = masks[~mat]


mat = np.random.choice(a=[False, True], size=(len(images)), p=[0.88, 0.12])

test_images = images[mat]
test_masks = masks[mat]

images = images[~mat]
masks = masks[~mat]

print(f'Train {len(images)}, Validation {len(validation_images)}, Test {len(test_images)}')

np.save('npy_datasets/cv_data/cv_train_images.npy', images)
np.save('npy_datasets/cv_data/cv_train_masks.npy', masks)

np.save('npy_datasets/cv_data/cv_val_images.npy', validation_images)
np.save('npy_datasets/cv_data/cv_val_masks.npy', validation_masks)

np.save('npy_datasets/cv_data/cv_test_images.npy', test_images)
np.save('npy_datasets/cv_data/cv_test_masks.npy', test_masks)


