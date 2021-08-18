from preprocessor import Preprocessor
from PIL import Image
import os


if __name__ == '__main__':

    source_path = './data_preview/raw_img/'
    filenames = os.listdir(source_path)

    for idx, name in enumerate(filenames):
        image = Image.open(source_path + name)
        p = Preprocessor(image)
        p.autopreprocess()
        p.save(name, path='./data_preview/processed_img/')

        print(f'Succesfully saved image number: {idx}')