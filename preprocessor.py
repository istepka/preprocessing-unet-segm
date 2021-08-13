from PIL import Image


class Preprocessor:
    '''Image preprocessing'''

    def __init__(self, img: Image) -> None:
        self.img = img

    def resize(self, target_resolution=(512,512)) -> None:
        '''Resize image to target resolution'''
        self.img = self.img.resize(target_resolution)

    def add_border(self, width=20) -> None:
        '''Add fixed-width border to image'''
        tmp = Image.new("RGB", self.img.size)

        shrinked_old_dim = self.img.size[0] - width*2
        self.resize(target_resolution=(shrinked_old_dim, shrinked_old_dim))
        
        tmp.paste(self.img, (width, width))
        self.img = tmp
    
    def convert_to_grayscale(self) -> None:
        '''Convert image to grayscale'''
        self.img = self.img.convert('L')

    def save(self, name, path="./data/temp/") -> None:
        self.img.save(path + name)


