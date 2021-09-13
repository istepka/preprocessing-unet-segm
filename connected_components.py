import cv2 
import numpy as np
from preprocessing.preprocessor import Preprocessor
from PIL import Image

path_to_image = 'src/data/raw_img/images/ISIC_0000043.jpg'

im = cv2.imread(path_to_image)
print(im.shape)
p = Preprocessor( Image.fromarray(im).convert(mode='L'))
p.convert_to_grayscale()
p.remove_vignette()
image =  p.get_processed_np_img(normalized=True)*255
image = image.flatten().reshape((image.shape[0], image.shape[1]))
image = np.array(image, dtype='uint8')

print(type(image), image.shape, image[400][600], image.max(), image.min())

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
print(type(gray), gray.shape, gray[400][600], gray.max(), gray.min())


thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


cv2.imshow('image', thresh)
cv2.waitKey(0)
    
outputt = cv2.connectedComponentsWithStats(thresh, connectivity=4, ltype=cv2.CV_32S)
(numLabels, labels, stats, centroids) = outputt

# loop over the number of unique connected component labels
for i in range(0, numLabels):
	# if this is the first component then we examine the
	# *background* (typically we would just ignore this
	# component in our loop)
	if i == 0:
		text = "examining component {}/{} (background)".format(
			i + 1, numLabels)
	# otherwise, we are examining an actual connected component
	else:
		text = "examining component {}/{}".format( i + 1, numLabels)
	# print a status message update for the current connected
	# component
	print("[INFO] {}".format(text))
	# extract the connected component statistics and centroid for
	# the current label
	x = stats[i, cv2.CC_STAT_LEFT]
	y = stats[i, cv2.CC_STAT_TOP]
	w = stats[i, cv2.CC_STAT_WIDTH]
	h = stats[i, cv2.CC_STAT_HEIGHT]
	area = stats[i, cv2.CC_STAT_AREA]
	(cX, cY) = centroids[i]

    

# clone our original image (so we can draw on it) and then draw
# a bounding box surrounding the connected component along with
# a circle corresponding to the centroid
output = image.copy()
cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
cv2.circle(output, (int(cX), int(cY)), 8, (0, 0, 255), -1)

# construct a mask for the current connected component by
# finding a pixels in the labels array that have the current
# connected component ID
componentMask = (labels == 0).astype("uint8") * 255
# show our output image and connected component mask
cv2.imshow("Output", output)
cv2.imshow("Connected Component 0", componentMask)
componentMask = (labels == 1).astype("uint8") * 255
cv2.imshow("Connected Component 1", componentMask)
cv2.waitKey(0)

unique, counts = np.unique(labels, return_counts=True)
print(sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[0:10])