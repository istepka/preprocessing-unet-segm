import numpy as np
from PIL import Image
import os
from data_loader import DataLoader


source_path_img = './src/data/raw_img/images/'
filenames = os.listdir(source_path_img)


dl = DataLoader()
images, masks = dl.get_dataset(resolution=512)

data = np.array((images, masks))

np.save('data_512p', data)
