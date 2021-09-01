import numpy as np
from PIL import Image
import os
from data_loader import DataLoader


source_path_img = './src/data/raw_img/images/'
filenames = os.listdir(source_path_img)


dl = DataLoader()
images, masks = dl.get_dataset(resolution=256, 
                                source_path_img='./src/data/validation/images/images/',
                                source_path_mask='./src/data/validation/masks/masks/'
                                    )

data = np.array((images, masks))

np.save('data_validation', data)
