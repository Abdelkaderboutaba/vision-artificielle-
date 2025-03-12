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

# Function to stitch two images progressively
def stitch_images(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Match keypoints using BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des2, des1, k=2)

    # Apply ratio test
    good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

    # Compute homography if enough matches are found
    MIN_MATCH_COUNT = 10
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Warp second image to align with first image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        warped_img = cv2.warpPerspective(img2, M, (w1 + w2, h1))
        
        # Blend images (simple overlay)
        warped_img[0:h1, 0:w1] = img1

        return warped_img
    else:
        print("Not enough matches found between images.")
        return None

# Load a list of images
image_paths = ['image0.jpg', 'image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg','image5.jpg']
images = [cv2.imread(path) for path in image_paths]

# Check if all images are loaded successfully
for i, img in enumerate(images):
    if img is None:
        print(f"Error: Unable to load image at index {i}.")
        exit()

# Stitch images progressively (pair by pair)
result = images[0]
for i in range(1, len(images)):
    print(f"Stitching image {i} with the previous result...")
    result = stitch_images(result, images[i])
    if result is None:
        print(f"Stitching failed at image {i}.")
        break

# Trim and save the final stitched image
if result is not None:
    result = trim(result)
    cv2.imshow("Final Stitched Image", result)
    cv2.waitKey(0)
    cv2.imwrite("final_stitched_image.jpg", result)
    print("Stitching completed and saved as 'final_stitched_image.jpg'.")
