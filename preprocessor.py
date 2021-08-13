from PIL import Image, ImageOps


class Preprocessor:
    '''Image preprocessing'''

    def __init__(self, img: Image) -> None:
        self.img = img

    def resize(self, target_resolution: tuple(int, int) =(512,512)) -> None:
        '''Resize image to target resolution'''
        self.img = self.img.resize(target_resolution)

    def add_border(self, width: int=20) -> None:
        '''Add fixed-width border to image'''
        tmp = Image.new("RGB", self.img.size)

        shrinked_old_dim = self.img.size[0] - width*2
        self.resize(target_resolution=(shrinked_old_dim, shrinked_old_dim))
        
        tmp.paste(self.img, (width, width))
        self.img = tmp
    
    def convert_to_grayscale(self) -> None:
        '''Convert image to grayscale'''
        self.img = self.img.convert('L')

    def enchance_contrast(self, cutoff_percentage: float =0.02) -> None:
        '''Enchance contrast by cut off higest & lowest histogram x%\n
        cutoff_percentage: Top and bottom percentage  of histogram intensities to be cut off. (0-1)
        '''
        ImageOps.autocontrast(self.img, cutoff=cutoff_percentage)


    def save(self, name: str, path: str="./data/temp/") -> None:
        self.img.save(path + name)


