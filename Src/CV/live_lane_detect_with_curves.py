import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2  # Import OpenCV
from moviepy.editor import VideoFileClip  # Import VideoFileClip

# Define the function to display images
def list_images(images, cols=2, rows=None, cmap=None, output_file='output_images.png'):
    num_images = len(images)
    if rows is None:
        rows = (num_images + cols - 1) // cols  # Ceiling division

    plt.figure(figsize=(10, 11))
    
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        cmap_to_use = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap=cmap_to_use)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    
    # Save the figure to a file
    plt.savefig(output_file)  # Specify the output filename
    plt.close()  # Close the figure to free memory

def convert_hsl(image):
    """Convert an image from RGB to HSL."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def HSL_color_selection(image):
    """Apply color selection to the HSL images to blackout everything except for white lane lines."""
    converted_image = convert_hsl(image)
    
    # White color mask
    lower_threshold = np.uint8([0, 200, 0])  # H, L, S
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)
    
    # Combine masks (only white in this case)
    masked_image = cv2.bitwise_and(image, image, mask=white_mask)
    
    return masked_image

def gray_scale(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussian_smoothing(image, kernel_size=5):   # kernel size was 5
    """Apply Gaussian smoothing to the image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def canny_detector(image, low_threshold=50, high_threshold=150):
    """Apply Canny edge detection to the image."""
    return cv2.Canny(image, low_threshold, high_threshold)


#DEBUG trying to redefined a bigger triangle 
def canny(image):
    if image is None:
        cap.release()
        cv2.destroyAllWindows()
        exit()
    gray = cv2.cvtColor(image,cv2.COLOR_BAYER_BG2BGR)
    kernel = 5
    blur =cv2.GaussianBlur(gray,(kernel,kernel),0)
    canny = cv2.Canny(gray,50,150)
    return canny


###DEBUG trying to redefined a bigger triangle Nestor 
def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    triangle = np.array([[
        (200,height),
        (800,350),
        (1200,height),]],np.int32)
    cv2.fillPoly(mask,triangle,255)
    masked_image = cv2.bitwise_and(canny,mask)
    return masked_image

def region_selection(image):
    """Determine and cut the region of interest in the input image."""
    mask = np.zeros_like(image)   
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    rows, cols = image.shape[:2]
    # bottom_left  = [cols * 0.01, rows * 0.9]  # col was 0.1
    # top_left     = [cols * 0.35, rows * 0.48]  #It was at cols * 0.2 and rows 0.4
    # bottom_right = [cols * 0.99, rows * 0.9]
    # top_right    = [cols * 0.75, rows * 0.48]  #cols 0.8 rows 0.4

    ##TESTING - DEBIN
    top_left     = [cols * 0.3, rows * 0.4]  #It was at cols * 0.2 and rows 0.4
    top_right    = [cols * 0.7, rows * 0.4]  #cols 0.8 rows 0.4
    bottom_left  = [cols * 0.1, rows * 0.9]  # col was 0.1
    bottom_right = [cols * 0.9, rows * 0.9]
    
    #defines the region
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

def hough_transform(image):
    """Determine lines in the image using the Hough Transform."""
    rho = 1              # Distance resolution of the accumulator in pixels.
    theta = np.pi / 180  # Angle resolution of the accumulator in radians.
    threshold = 20       # Only lines that are greater than threshold will be returned. Higher the number, fewer lines detected
    minLineLength = 20   # Line segments shorter than that are rejected. Minimum length of line to be detected
    maxLineGap = 300     # Maximum allowed gap between points on the same line to link them. Max gap allowed betweeen the same line to be detected
    lines = cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                             minLineLength=minLineLength, maxLineGap=maxLineGap)
    return lines if lines is not None else []

def average_slope_intercept(lines):
    """Find the slope and intercept of the left and right lanes of each image."""
    if len(lines) == 0:  # Check if lines is empty
        return None, None  # Return None if no lines are detected

    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:  # Skip vertical lines
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def pixel_points(y1, y2, line):
    """Converts the slope and intercept of each line into pixel points."""
    if line is None:
        return None
    slope, intercept = line
    
    if slope == 0:  # Prevent division by zero
        return None
    
    # Calculate x1 and x2 for the given y1 and y2
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return ((x1, int(y1)), (x2, int(y2)))


def lane_lines(image, lines):
    """Create full length lines from pixel points."""
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    
    return left_line, right_line


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
    """Draw lines onto the input image."""
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def frame_processor(image):
    """
    Process the input frame to detect lane lines.
        Parameters:
            image: Single video frame.
    """
    color_select = HSL_color_selection(image)
    gray = gray_scale(color_select)
    smooth = gaussian_smoothing(gray)
    edges = canny_detector(smooth)
    region = region_selection(edges)

    cv2.imshow('Region of interest with 1st triangle', region)
    
    hough = hough_transform(region)
    result = draw_lane_lines(image, lane_lines(image, hough))
    
    return result

def webcam_video_processing():
    """Capture video from the webcam and process it for lane detection."""
    cap = cv2.VideoCapture("/dev/video0")  # Use 0 for the default webcam
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame = frame_processor(frame)

        #Display resolution of camera for debug- Debin
        # CamWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # CamHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(f"Camera Resolution: {CamWidth}x{CamHeight}")

        # Display the resulting frame
        cv2.imshow('Lane Detection - White Lines Only', processed_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the webcam processing
webcam_video_processing()
