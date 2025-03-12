#reference: https://pylessons.com/OpenCV-image-stiching-continue

import cv2
import numpy as np

#original_image_left
#img_l = cv2.imread('image0.jpg')
img_l = cv2.imread('img1.jpg')
imgl = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)

#original_image_right
img_r = cv2.imread('img2.jpg')
#img_r = cv2.imread('image1.jpg')
imgr = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
# find key points
kpl, desl = sift.detectAndCompute(imgl,None)
kpr, desr = sift.detectAndCompute(imgr,None)

cv2.imshow('original_image_left_keypoints',cv2.drawKeypoints(img_l,kpl,None))
cv2.waitKey(0)

cv2.imshow('original_image_right_keypoints',cv2.drawKeypoints(img_r,kpr,None))
cv2.waitKey(0)

match = cv2.BFMatcher()
matches = match.knnMatch(desr,desl,k=2)

good = []
for m,n in matches:
    if m.distance < 0.3*n.distance:
        good.append(m)

draw_params = dict(matchColor=(0,255,0),
                       singlePointColor=None,
                       flags=2)

img3 = cv2.drawMatches(img_r,kpr,img_l,kpl,good,None,**draw_params)
cv2.imshow("Draw Matches Left Right.jpg", img3)
cv2.waitKey(0)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kpr[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpl[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print(M)

    h,w = imgr.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
else:
    print("Not enought matches are found - %d/%d", (len(good)/MIN_MATCH_COUNT))

print(img_l.shape[1] + img_r.shape[1], img_l.shape[0])

dst = cv2.warpPerspective(img_r,M,(img_l.shape[1] + img_r.shape[1], img_l.shape[0]))
cv2.imshow("dst warpPerspective", dst)
cv2.waitKey(0)

dst[0:img_l.shape[0],0:img_l.shape[1]] = img_l
cv2.imshow("add left image to the warped right.jpg", dst)
cv2.waitKey(0)

def trim(frame):
    #crop 
    if not np.sum(frame[0]):
        print('1')
        return trim(frame[1:])
    #crop 
    if not np.sum(frame[-1]):
        print('2')
        return trim(frame[0:-1])
    #crop 
    if not np.sum(frame[:,0]):
        print('3')
        return trim(frame[:,1:])
    #crop 
    if not np.sum(frame[:,-1]):
        print('4')
        return trim(frame[:,:-2])
    return frame

cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
cv2.waitKey(0)
cv2.imwrite("original_image_stitched_crop.jpg", trim(dst))
