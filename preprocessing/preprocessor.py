from typing import Tuple
from PIL import Image, ImageFilter, ImageOps, ImageStat
import numpy as np
import sys, os


class Preprocessor:
    '''Image preprocessing'''

    def __init__(self, img: Image) -> None:
        self.img = img.copy()
        self.raw_img = img.copy()

    def autopreprocess(self) -> None:
        '''Execute whole preprocessing sequence over an image'''
        self.convert_to_grayscale()
        self.enchance_contrast()
        self.remove_vignette()
        self.remove_hair()
        self.resize()
        self.add_border()

    def resize(self, target_resolution: Tuple[int, int] = (512,512)) -> None:
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
        self.img = ImageOps.autocontrast(self.img, cutoff=cutoff_percentage)

    def remove_hair(self) -> None:
        '''Remove hair from image by blurring detected boundaries'''
        #im_edges = self.img.filter(ImageFilter.FIND_EDGES)
        #TODO
        pass
    
    def remove_vignette(self) -> None:
        '''Remove vignette effect from image and replace its area by neutral pixels'''
        source_image = self.img.copy()
        #source_image = source_image.crop(box= (20,20, 492, 492) )
        #TODO Make sure that we operate on the non-resized and non-border version 

        height = source_image.size[0]
        width = source_image.size[1]
        center = (width/2, height/2)

        for iteration in range(22):
            #print(f"Iteration: {iteration}", file=sys.stderr, flush=True)
            radius = height / 3 - 20 + iteration * 15

            #Create circular mask
            Y, X = np.ogrid[:width, :height]
            dist_from_center = np.sqrt((X - center[1])**2 + (Y-center[0])**2)
            arr_mask = (dist_from_center <= radius)*255
            
            #Apply mask to source image 
            im_mask = Image.fromarray(arr_mask).convert('L')
            #masked_img = Image.composite(source_image, im_mask, mask=im_mask)

            #Get mean pixel values of inner and outer circle
            mean_inner = ImageStat.Stat(source_image, mask=im_mask).mean[0]
            mean_outer = ImageStat.Stat(source_image, mask=ImageOps.invert(im_mask)).mean[0]
            

            #Check if vignette effect is in the image
            if mean_outer < 30.0:
                print(f'inner: {mean_inner} \nouter:{mean_outer}\n-------')
                inner_fill = np.full((width, height), int(mean_inner))
                inner_fill = Image.fromarray(inner_fill).convert('L')

                corrected_img = Image.composite(source_image, inner_fill,mask=im_mask)
                source_image.paste(corrected_img)
                
                self.img = source_image
                break 
    

    def save(self, name: str, path: str="./src/data/temp/") -> None:
        if not os.path.exists(path):
            os.makedirs(path)
        self.img.save(path + name)


