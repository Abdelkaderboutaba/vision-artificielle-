import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('scene2.png',0)
imgR = cv2.imread('scene3.png',0)

stereo = cv2.StereoSGBM_create(numDisparities=21,blockSize=16)
# StereoSGBM is the implementation of Hirschmüller’s original SGM [2] algorithm. 
# SGBM stands for Semi-Global Block Matching. It also implements the sub-pixel 
#estimation proposed by Brichfield et al. [3]
#[2] Hirschmüller, Heiko (2005). “Accurate and efficient stereo processing 
#by semi-global matching and mutual information”. 
#IEEE Conference on Computer Vision and Pattern Recognition. pp. 807–814.

#[3] . Birchfield and C. Tomasi, “Depth discontinuities by pixel-to-pixelstereo,
#”International Journal of Computer Vision, vol. 35, no. 3,pp. 269–293, 1999.

disparity = stereo.compute(imgL, imgR)
print(disparity)

plt.imshow(disparity,'gray')
#plt.imshow(disparity,'gray')
plt.show()