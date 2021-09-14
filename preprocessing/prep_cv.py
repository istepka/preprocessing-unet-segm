from functools import reduce
import cv2
from matplotlib.pyplot import imread 
import numpy as np
from six import b
from tensorflow.keras import backend as K


def resize(image, dimensions):
    assert str(image.dtype) == 'uint8', 'Array dtype should be uint8'
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

def grayscale_convertion(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def histogram_equalization(image, cutoff=0.02):
    return cv2.equalizeHist(image)

def gaussian_blur(image, radius=3):
    return cv2.blur(image, (radius, radius))

def connected_components(image, take=5, return_only_best=False, debug=False):

    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    out = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=cv2.CV_32S)

    (numLabels, labels, stats, centroids) = out
        
    unique, counts = np.unique(labels, return_counts=True)
    most_popular_labels = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[0:take]

    if debug:
        print('Most popular components', most_popular_labels)

    component_masks = list()

    for i,_ in most_popular_labels:
        if debug:
            if i == 0:
                text = "examining component {}/{} (background)".format(i + 1, numLabels)
            else:
                text = "examining component {}/{}".format( i + 1, numLabels)
            print("[INFO] {}".format(text))
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        output = image.copy()
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

        componentMask = (labels == i).astype("uint8") * 255

        if debug:
            cv2.imshow("Output", output)
            cv2.imshow("Connected Component", componentMask)
            cv2.waitKey(0)

        if return_only_best and i == 0:
            return componentMask

        component_masks.append(componentMask)
    
    if len(component_masks) < take: 
        empties = take - len(component_masks)
        if debug:
            print(f'Adding {empties} empty masks')

        for i in range(empties):
            component_masks.append(np.zeros( (image.shape[0], image.shape[1]), dtype='uint8'))
            
    return np.array(component_masks)

def connected_components_on_batch(batch_of_images, take=5, return_only_best=False, debug=False):
    '''Get connected components for images in a batch. Results will be added as new channels to the input batch'''
    print(type(batch_of_images))
    print(batch_of_images.shape)

    batch_of_images = np.array(batch_of_images, dtype='uint8')

    #If batch contains only one image then operations can be simplified
    if len(batch_of_images) == 1:
        return connected_components(batch_of_images, take=take, return_only_best=return_only_best, debug=debug)

    #Otherwise connected components masks are acquired for each image and are added as new dimesions to original image array
    cc = list()

    for i in range(len(batch_of_images)):
        cc.append(connected_components(batch_of_images[i], take=take, return_only_best=return_only_best, debug=debug))
    print('Gathered connected components masks')
        
    cc = np.array(cc, dtype='f')
    dim = batch_of_images[i].shape[0]
    new_batch = list()
    #Adding cc masks as new 'channel color' dimensions 
    for i, components_batch in enumerate(cc):
        new_image = batch_of_images[i]
        for cmask in components_batch:
            cmask = np.reshape(cmask, (dim,dim,-1))
            new_image = np.concatenate((new_image, cmask), axis=2)
        new_batch.append(new_image)
            

    print(f'New batch dims {np.array(new_batch).shape}')
    return np.array(new_batch)

def plot(image, mask):
    import matplotlib.pyplot as plt
    #res = (image > 0.25).astype(float)
    res = image

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1,2,1)
    ax.set_title('Ground truth')
    ax.imshow(mask, cmap='gray')


    ax1 = fig.add_subplot(1,2,2)
    ax1.set_title('Predicted mask')
    ax1.imshow(res, cmap='gray')

    plt.show()

if __name__ == '__main__':



    im  = imread('src/data/raw_img/images/ISIC_0000006.jpg')
    im2 = imread('src/data/raw_img/images/ISIC_0000007.jpg')

   
    
    c = cv2.resize(im, (256,256))
    c = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
    c = np.reshape(c, (256,256,-1))

    cc = cv2.resize(im2, (256,256))
    cc = cv2.cvtColor(cc, cv2.COLOR_RGB2GRAY)
    cc = np.reshape(cc, (256,256,-1))

    ccc = np.array([c, cc])
    #ccc = np.concatenate((c,cc), axis=0)
    print(ccc.shape)

    out = connected_components_on_batch(ccc)

    print(out.shape)

    cv2.imshow("Connected Component", c[:,:,:4])
    cv2.waitKey(0)