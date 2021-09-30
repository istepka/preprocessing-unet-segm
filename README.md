# Lesion skin cancer segmentation
### Based on the ISIC 2018 challenge which goal was to create automated predictions of lesion segmentation boundaries within dermoscopic images.
![img](src/overview_readme.png)
***

## Image segmentation  
Segmentation is performed using U-Net Convolutional Neural Network. 

### Best results on the test set are: Accuracy 95.1%, Jaccard score 80.8%, Sensitivity 86.8%, Specifitivity 97.4%
With parameters:
- image size: 256x256
- batch size: 8
- epochs: 70 (Early Stopping patience=10)
- feature channels configuration: [32,64,128,256,512]
- optimizer: adam
- loss: binary_crossentropy
- preprocessing: [augumentations, histogram equalization, per-channel normalization]

***

## Image Pre-Processing
### Challenges:
* vignette
* ink markings
* scale rulers
* skin lines
* blood vessels
* hair

### Common techniques
* color space transformation, grayscale convertion
* contrast enchancement, histogram equalization
* per-channel mean normalization
* gaussian blur
* data augumentation (horiz/vert flip, zoom, rotation, offset)
* ZCA
* connected components


### Pre-processings currently implemented
1. Resize image to a constant computable dimension (e.g. 512x512)  
1. Convert image to grayscale to reduce image dimension and match format of orginal U-Net reference  
1. Contrast enchancement  
    This ensures images have consistent contrast between neighbouring areas and RoI. It will be accomplished by first calculating the histogram of grayscale image. The top 2% of histogram intensities were then selected and used as cut off values. Histogram then stretches to remap the darkest pixel to 0 and lightest to 255 against the selected cut off thresholds.  
1. Per-channel mean normalization
1. Data augumentation 
1. ZCA whitening
1. Connected components




***
## References  
1. Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab N., Hornegger J., Wells W., Frangi A. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science, vol 9351. Springer, Cham. https://doi.org/10.1007/978-3-319-24574-4_28
  
1. Sara Ross-Howe, H.R. Tizhoosh - "The Effects of Image Pre- and Post-Processing, Wavelet Decomposition, and Local Binary Patterns on U-Nets for Skin Lesion Segmentation" - Kimia Lab, University of Waterloo, Waterloo, Ontario, Canada  

1. Tschandl, P. et al. The HAM10000 dataset, a large collection of multi-source
dermatoscopic images of common pigmented skin lesions. Sci. Data 5:180161 doi: 10.1038/sdata.2018.161 (2018).  

1. Nabila Abraham, Naimul Mefraz Khan - "A NOVEL FOCAL TVERSKY LOSS FUNCTION WITH IMPROVED ATTENTION U-NET FOR LESION SEGMENTATION" - Ryerson University Department of Electrical and Computer Engineering 350 Victoria Street, Toronto, ON  

1. https://challenge2018.isic-archive.com/participate/

