import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import p1
import math

video_path = r'test_videos\challenge.mp4'
file_name = 'edited_challenge.avi'

def mean(list):
    """
    calculate mean of list
    """
    return float(sum(list)) / max(len(list), 1)

def draw_lines(img, lines, color=[255, 0, 0], thickness=12):
    # initialize lists to hold line formula values
    bLeftValues     = []  # b of left lines
    bRightValues    = []  # b of Right lines
    mPositiveValues = []  # m of Left lines
    mNegitiveValues = []  # m of Right lines
    
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                
                # calculate slope and intercept
                m = (y2-y1)/(x2-x1)
                b = y1 - x1*m
                
                # threshold to check for outliers
                if m >= 0 and (m < 0.2 or m > 0.8):
                    continue
                elif m < 0 and (m < -0.8 or m > -0.2):
                    continue
                    
                # seperate positive line and negative line slopes
                if m > 0:
                    mPositiveValues.append(m)
                    bLeftValues.append(b)
                else:
                    mNegitiveValues.append(m)
                    bRightValues.append(b)
        
    # Get image shape and define y region of interest value
    imshape = img.shape
    y_max   = imshape[0] # lines initial point at bottom of image    
    y_min   = y_max/5*3        # lines end point at top of ROI

    # Get the mean of all the lines values
    AvgPositiveM = mean(mPositiveValues)
    AvgNegitiveM = mean(mNegitiveValues)
    AvgLeftB     = mean(bLeftValues)
    AvgRightB    = mean(bRightValues)
    x1_Left=1
    y1_Left=1
    x2_Left=1
    y2_Left=1
    # use average slopes to generate line using ROI endpoints
    if AvgPositiveM != 0:
        x1_Left = (y_max - AvgLeftB)/AvgPositiveM
        y1_Left = y_max
        x2_Left = (y_min - AvgLeftB)/AvgPositiveM
        y2_Left = y_min
    if AvgNegitiveM != 0:
        x1_Right = (y_max - AvgRightB)/AvgNegitiveM
        y1_Right = y_max
        x2_Right = (y_min - AvgRightB)/AvgNegitiveM
        y2_Right = y_min

        # define average left and right lines
        cv2.line(img, (int(x1_Left), int(y1_Left)), (int(x2_Left), int(y2_Left)), color, thickness) #avg Left Line
        cv2.line(img, (int(x1_Right), int(y1_Right)), (int(x2_Right), int(y2_Right)), color, thickness) #avg Right Line


# read video
video = cv2.VideoCapture(video_path)
# video output
codec = cv2.VideoWriter_fourcc(*'XVID') #MPEG 4 codec
fps = video.get(cv2.CAP_PROP_FPS)
video_size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
out = cv2.VideoWriter(file_name, codec, fps, video_size)
total_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0
print('total_frame:',total_frame)

while current_frame < total_frame:
    success, orig_frame = video.read()
    if not success:
        video = cv2.VideoCapture(video_path)
        continue

    gray = cv2.cvtColor(orig_frame,cv2.COLOR_RGB2GRAY)
    height = orig_frame.shape[0]
    width = orig_frame.shape[1]
    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    # sobel x
    sobelXImage = np.uint8(np.absolute(
                            cv2.Sobel(blur_gray, cv2.CV_64F, 1, 0)
                            ))

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    # edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    edges = cv2.Canny(sobelXImage, low_threshold, high_threshold)

    # region setting
    region_of_interest_vertices = np.array([[
        [0, height], [width/15*7, height/5*3],
        [width/15*8, height/5*3],[width, height]]],
        dtype = np.int32)

    # region_of_interest_vertices = np.array([[
    #     [0, height], [450, 330],
    #     [520, 330],[width, height]]],
    #     dtype = np.int32)

    # region interest
    after_region_line_image = p1.region_of_interest(edges, region_of_interest_vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1
    theta = np.pi/180
    threshold = 50 #can use to select the lines we need, more large than less line
    min_line_length = 40 # Minimum length of line. Line segments shorter than this are rejected.
    max_line_gap = 20 # Maximum allowed gap between line segments to treat them as single line.
    line_image = np.copy(orig_frame)*0 #creating a blank to draw lines on

    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(after_region_line_image, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # Iterate over the output "lines" and draw lines on the blank   
    draw_lines(line_image,lines,(0,0,255),10)
    
    # # Iterate over the output "lines" and draw lines on the blank
    # if lines is not None:
    #     for line in lines:
    #         for x1,y1,x2,y2 in line:
    #             cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),3)

    # Draw the lines on the edge image
    edited_frame = cv2.addWeighted(orig_frame, 0.8, line_image, 1, 0) 
    cv2.imshow("final", edited_frame)
    out.write(edited_frame)
    current_frame += 1

    key = cv2.waitKey(1)
    if key == 27: # press esc
        break
cv2.destroyAllWindows()
video.release()
out.release()