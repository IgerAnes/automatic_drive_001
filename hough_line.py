import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import p1


# Read in and grayscale the image
image = mpimg.imread(r'test_images\exit-ramp.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
height = image.shape[0]
width = image.shape[1]

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# sobel x
sobelXImage = np.uint8(np.absolute(
                        cv2.Sobel(blur_gray, cv2.CV_64F, 1, 0)
                        ))
cv2.imshow("sobel",sobelXImage)
# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(sobelXImage, low_threshold, high_threshold)
cv2.imshow("edges",edges)

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 1
theta = np.pi/180
threshold = 100 #can use to select the lines we need, more large than less line
min_line_length = 100 # Minimum length of line. Line segments shorter than this are rejected.
max_line_gap = 25 # Maximum allowed gap between line segments to treat them as single line.
line_image = np.copy(image)*0 #creating a blank to draw lines on

# Run Hough on edge detected image
lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                        min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on the blank
if lines is not None:
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

# region setting
region_of_interest_vertices = np.array([[
    [0, height], [width/3,height/2],
    [width/3*2,height/2],[width, height]]],
    dtype = np.int32)

# region interest
after_region_line_image = p1.region_of_interest(line_image, region_of_interest_vertices)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges)) 

# Draw the lines on the edge image
combo = cv2.addWeighted(image, 0.8, after_region_line_image, 1, 0) 
plt.imshow(combo)
plt.show()