import cv2
import numpy as np

# Function to trim black borders from the stitched image
def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-1])
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-1])
    return frame

# Function to stitch two images
def stitch_images(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Match keypoints using BFMatcher
    match = cv2.BFMatcher()
    matches = match.knnMatch(des2, des1, k=2)

    # Apply ratio test to filter good matches
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    # Compute homography if enough matches are found
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = gray2.shape

        # Warp the second image
        warped_img = cv2.warpPerspective(img2, M, (img1.shape[1] + img2.shape[1], img1.shape[0]))

        # Combine the warped image with the first image
        warped_img[0:img1.shape[0], 0:img1.shape[1]] = img1
        return warped_img
    else:
        print("Not enough matches found between images.")
        return None

# Load a list of images
image_paths1 = [ 'image0.jpg', 'image1.jpg', 'image2.jpg',]
images1 = [cv2.imread(path) for path in image_paths1]

image_paths2 = [ 'image3.jpg', 'image4.jpg','image5.jpg']
images2 = [cv2.imread(path) for path in image_paths2]

# Check if all images are loaded successfully
for i, img in enumerate(images1):
    if img is None:
        print(f"Error: Unable to load image at index {i}.")
        exit()

for i, img in enumerate(images2):
    if img is None:
        print(f"Error: Unable to load image at index {i}.")
        exit()

# Stitch images iteratively
result1 = images1[0]  # Start with the first image
for i in range(1, len(images1)):
    print(f"Stitching image {i+1}...")
    result1 = stitch_images(result1, images1[i])
    if result1 is None:
        print(f"Stitching failed at image {i+1}.")
        break

result2 = images2[0]  # Start with the first image
for i in range(1, len(images2)):
    print(f"Stitching image {i+1}...")
    result2 = stitch_images(result2, images2[i])
    if result2 is None:
        print(f"Stitching failed at image {i+1}.")
        break

resultf = stitch_images(result1, result2)

cv2.imshow("Final Stitched Image", trim(resultf))
# cv2.imshow("Final Stitched Image", trim(result2))

# Trim and save the final stitched image
if resultf is not None:
    result = trim(resultf)
    
    # cv2.imshow("Final Stitched Image", result)
    cv2.waitKey(0)
    cv2.imwrite("final_stitched_image.jpg", result)
    print("Stitching completed and saved as 'final_stitched_image.jpg'.")