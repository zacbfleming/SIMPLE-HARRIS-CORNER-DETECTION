import numpy as np
import cv2
from matplotlib import pyplot as plt

simA = cv2.imread('/check.bmp')
transA = cv2.imread('/check_rot.bmp')

###Add gaussian blur
def AddG(img):
    GB = cv2.GaussianBlur(img, (3,3), 9, 9)
    return GB

### apply xy sobel gradients
def xygrad(img):
    im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    npimg = np.asarray(im[0:np.shape(im)[0], 0:np.shape(im)[1]])
    sobelx = cv2.Sobel(npimg,cv2.CV_32F,1,0,ksize=9)
    sobely = cv2.Sobel(npimg,cv2.CV_32F,0,1,ksize =9)
    return sobelx, sobely

### Extract features from image using harris corrner detector
def H(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    i = np.float32(image)
    im = cv2.cornerHarris(i, 4, 1, 0.1)
    im = cv2.dilate(im,None)
    dst2 = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    dst2[im>0.1*im.max()] = [255,0,0]
    return dst2
    

img = AddG(simA)
img2 = AddG(transA)
cv2.imshow('simA with Gaussian Blur', img)
cv2.imshow('transA with Gaussian Blur', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
i = xygrad(img)
i2 = xygrad(img2)
cv2.imshow('SimA Sobel X', i[0])
cv2.imshow('SimA Sobel Y', i[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('transA Sobel X', i2[0])
cv2.imshow('transA Sobel Y', i2[1])
cv2.waitKey(0)
cv2.destroyAllWindows()
image = H(simA)
image2 = H(transA)
cv2.imshow('simA', image)
cv2.imshow('transA', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()
