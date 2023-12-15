import cv2
import numpy as np
import matplotlib.image as mpimg

# Load the image
img = cv2.imread(r'D:\4th Year File (7th Semester)\AER 850\Project3\Project 3 Data\data\motherboard_image.jpeg')

# Resizing image
img = cv2.resize(img, (1448,1086))

# Setting the image to gray
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_ , thresh = cv2.threshold(gray,np.mean(gray), 255, cv2.THRESH_BINARY_INV)

# Contours
contours , hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cont = sorted(contours, key=cv2.contourArea)[-1]

# Edge detection
edges = cv2.Canny(img, threshold1=180,threshold2=200)

# Masking
mask = np.zeros(edges.shape, dtype="uint8" )
maskedFinal = cv2.drawContours(mask,[cont] , -1 , (255 , 255 , 255), -1)

# Extracting the motherboard from the original image
finalImage = cv2.bitwise_and(img, img, mask=maskedFinal)

# Showing the Result
cv2.imshow("Original", img)
cv2.imshow("Mask", mask)
cv2.imshow("Edge Detection", edges)
cv2.imshow("MaskedFinal", finalImage)
cv2.waitKey(0)


