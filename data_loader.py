from PIL import Image
import os
import numpy as np


class DataLoader:

    def __init__(self) -> None:
        self.images = list()
        self.masks = list()
        

    def get_dataset(self, resolution=512, n=100):
        source_path_img = './src/data/raw_img/'
        source_path_mask = './src/data/raw_masks/'

        filenames = list(map(lambda x: x[:-4] ,os.listdir(source_path_img)))[0:n+1]

       

        for i, name in enumerate(filenames):
            image = Image.open(source_path_img + name + '.jpg').convert(mode='L').resize((resolution,resolution))
            mask = Image.open(source_path_mask + name + '_segmentation.png').convert(mode='L').resize((resolution,resolution))

            image = np.asarray(image.getdata()).reshape(image.size[1], image.size[0], -1) 
            mask = np.asarray(mask.getdata()).reshape(mask.size[1], mask.size[0], -1)

            #image, mask = self.__normalize(image, mask)

            self.images.append(image)
            self.masks.append(mask)

            if i % 100 == 0:
                print(f'Loaded {i+1} images.')

        
      
        return np.array(self.images), np.array(self.masks)
        

if __name__ == "__main__":
    x,y = DataLoader().get_dataset()
    x = x/255
    print(x[0])