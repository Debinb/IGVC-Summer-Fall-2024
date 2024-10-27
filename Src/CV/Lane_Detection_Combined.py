# Import necessary libraries
import cv2 
import numpy as np
import math

# Function to preprocess the image to detect yellow and white lanes
def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gblur = cv2.GaussianBlur(gray, (5, 5), 0)
    white_mask = cv2.threshold(gblur, 200, 255, cv2.THRESH_BINARY)[1]
    lower_yellow = np.array([0, 100, 100])
    upper_yellow = np.array([210, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Apply morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask_cleaned

# Function that defines the polygon region of interest
def regionOfInterest(img, polygon):
    mask = np.zeros_like(img)
    pts = np.array(polygon, np.int32)
    cv2.fillPoly(mask, [pts], 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

# Function to warp the image for curved lane detection
def warp(img, source_points, destination_points, destn_size):
    matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    warped_img = cv2.warpPerspective(img, matrix, destn_size)
    return warped_img

# Function to fit curves to the lanes
def fitCurve(img):
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 50
    margin = 100
    minpix = 50
    window_height = int(img.shape[0] / nwindows)
    y, x = img.nonzero()
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_indices = []
    right_lane_indices = []

    for window in range(nwindows):
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_indices = ((y >= win_y_low) & (y < win_y_high) & (x >= win_xleft_low) & (x < win_xleft_high)).nonzero()[0]
        good_right_indices = ((y >= win_y_low) & (y < win_y_high) & (x >= win_xright_low) & (x < win_xright_high)).nonzero()[0]
        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)
        if len(good_left_indices) > minpix:
            leftx_current = int(np.mean(x[good_left_indices]))
        if len(good_right_indices) > minpix:
            rightx_current = int(np.mean(x[good_right_indices]))

    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)
    leftx = x[left_lane_indices]
    lefty = y[left_lane_indices]
    rightx = x[right_lane_indices]
    righty = y[right_lane_indices]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

# Function to calculate the radius of curvature using the warped image
def calculateCurvature(left_fit, right_fit, img_shape):
    y_eval = img_shape[0] - 1  # Evaluate at the bottom of the image
    left_curvature = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.abs(2 * left_fit[0])
    right_curvature = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.abs(2 * right_fit[0])
    avg_curvature = (left_curvature + right_curvature) / 2
    return avg_curvature

# Function that determines whether the lane is straight or curved using the warped image
def classifyLane(left_fit, right_fit, warped_img_shape):
    curvature = calculateCurvature(left_fit, right_fit, warped_img_shape)
    # Classification based on curvature radius: straight if high curvature, curved if lower curvature
    if curvature > 2000:  # Adjust this threshold as necessary
        return 'straight'
    else:
        return 'curved'

# Function to draw straight lines on the image
def drawStraightLines(img, masked_img):
    lines = cv2.HoughLinesP(masked_img, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return img

# Main video processing loop for real-time display
video = cv2.VideoCapture("/dev/video0")

while True:
    isTrue, frame = video.read()
    if not isTrue:
        break
    
    processed_img = preprocessing(frame)
    height, width = processed_img.shape
    polygon = [(int(width * 0.01), height), (int(width * 0.15), int(height * 0.55)), (int(width * 0.85), int(height * 0.55)), (int(width * 0.99), height)]
    masked_img = regionOfInterest(processed_img, polygon)
    cv2.imshow("Region of Interest", masked_img)
    # Define source and destination points for warping
    source_points = np.float32([[int(width * 0.01), int(height * 0.5)], [int(width * 0.98), int(height * 0.62)], [int(width * 0.05), height], [int(0.99 * width), height]])
    destination_points = np.float32([[0, 0], [400, 0], [0, 960], [400, 960]])
    warped_img_size = (400, 960)
   
    # Warp the image to get bird's-eye view
    warped_img = warp(masked_img, source_points, destination_points, warped_img_size)
    cv2.imshow("Warped",warped_img)

    # Fit curves to lanes in the warped image
    left_fit, right_fit = fitCurve(warped_img)

    # Classify lane type using the warped image
    lane_type = classifyLane(left_fit, right_fit, warped_img.shape)
    print(lane_type)

    if lane_type == 'straight':
        result_frame = drawStraightLines(frame, masked_img)
    else:
        result_frame = frame  # Add your visualization for curved lines here, similar to straight line handling

    # Display the result frame in a window
    cv2.imshow('Lane Detection', result_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
