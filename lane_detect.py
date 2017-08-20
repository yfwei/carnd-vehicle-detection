import numpy as np
import cv2

# Define conversions in x and y from pixels space to meters
# The values are derived from the warped image.
ym_per_pix = 12 / 720 # meters per pixel in y dimension
xm_per_pix = 3.7 / 880 # meters per pixel in x dimension

# Moving average running number
MOVING_AVERAGE_N = 5

# The source points for the perspective transform
perspective_transform_src = np.float32(
        [[555, 475],
         [730, 475],
         [1060, 680],
         [240, 680]])

# The width of the windows +/- margin
margin = 100

# White color threshold
WHITE_THRESHOLD = 185

# Yellow color threshold
YELLOW_THRESHOLD = 150


def white_thresh(img, thresh=128):
    color_channel = img[:,:,2]
    binary_output = np.zeros_like(color_channel)
    binary_output[color_channel > thresh] = 1

    return binary_output


def yellow_thresh(img, thresh=128):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel,a_channel,b_channel = cv2.split(lab)
    binary_output = np.zeros_like(b_channel)
    binary_output[b_channel > thresh] = 1

    return binary_output


def thresholded_binary_image(img):
    # Search white color
    white_binary = white_thresh(img, thresh=WHITE_THRESHOLD)

    # Search yellow color
    yellow_binary = yellow_thresh(img, thresh=YELLOW_THRESHOLD)

    # Combine the yellow and white filtered result
    combined = np.zeros_like(yellow_binary)
    combined[((white_binary == 1) | (yellow_binary == 1))] = 1

    return combined, white_binary, yellow_binary


def warp_image(img):
    offset = 200 # offset for dst points
    img_size = (img.shape[1], img.shape[0])

    dst = np.float32([[offset, 0], [img_size[0]-offset, 0],
                                     [img_size[0]-offset, img_size[1]],
                                     [offset, img_size[1]]])

    M = cv2.getPerspectiveTransform(perspective_transform_src, dst)
    Minv = cv2.getPerspectiveTransform(dst, perspective_transform_src)
    warped = cv2.warpPerspective(img, M, img_size)

    return warped, Minv

def lane_curvature(fity, left_fitx, right_fitx):
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(fity*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(fity*ym_per_pix, right_fitx*xm_per_pix, 2)

    # Compute the radius of curvature at the bottom of the image
    y_eval = np.max(fity)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad


def vehicle_pos(img, left_fitx, right_fitx):
    # The midpoint at the bottom of the image between the left and the right line
    camera_center = (left_fitx[-1] + right_fitx[-1]) // 2

    return (img.shape[1] // 2 - camera_center) * xm_per_pix


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #polynomial coefficients of the last n fits of the line
        self.recent_fits = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None


def find_lines_full(warped, margin):
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(warped.shape[0]/nwindows)

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    return left_lane_inds, right_lane_inds


def is_good_lines(img, left_line, right_line):
    # The left line and the right line should be roughly parallel
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    leftx = left_line.current_fit[0]*ploty**2 + left_line.current_fit[1]*ploty + left_line.current_fit[2]
    rightx = right_line.current_fit[0]*ploty**2 + right_line.current_fit[1]*ploty + right_line.current_fit[2]

    lane_width = (rightx[-1] - leftx[-1])
    diff = np.mean(rightx - leftx)
    if abs(lane_width - diff) > 150:
        return False

    # The curvature should be similar
    if left_line.current_fit[0] > 0 and right_line.current_fit[0] < 0:
        return False

    if left_line.current_fit[0] < 0 and right_line.current_fit[0] > 0:
        return False

    return True;


def find_lines(warped, left_line, right_line):
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if left_line.detected and right_line.detected:
        left_fit = left_line.best_fit
        right_fit = right_line.best_fit

        # Search from the last lane center
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
    else:
        left_lane_inds, right_lane_inds = find_lines_full(warped, margin)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_line.detected = (leftx.size > 0)
    right_line.detected = (rightx.size > 0)
    if left_line.detected is False or right_line.detected is False:
        return

    # Fit a second order polynomial to each
    right_fit = np.polyfit(righty, rightx, 2)
    left_fit = np.polyfit(lefty, leftx, 2)

    left_line.current_fit = left_fit
    right_line.current_fit = right_fit

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_line.allx = leftx
    right_line.allx = rightx
    left_line.ally = lefty
    right_line.ally = righty

    if is_good_lines(warped, left_line, right_line) is False:
        if len(left_line.recent_fits) == 0:
            left_line.detected = False

        if len(right_line.recent_fits) == 0:
            right_line.detected = False

        return

    left_line.recent_fits.append(left_fit)
    right_line.recent_fits.append(right_fit)

    # keep the last 5 measurements
    if len(left_line.recent_fits) > MOVING_AVERAGE_N:
        left_line.recent_fits.pop(0)

    if len(right_line.recent_fits) > MOVING_AVERAGE_N:
        right_line.recent_fits.pop(0)

    # Average the measurements
    left_line.best_fit = np.mean(np.asarray(left_line.recent_fits), axis=0)
    right_line.best_fit = np.mean(np.asarray(right_line.recent_fits), axis=0)

    left_line.bestx = left_line.best_fit[0]*ploty**2 + left_line.best_fit[1]*ploty + left_line.best_fit[2]
    right_line.bestx = right_line.best_fit[0]*ploty**2 + right_line.best_fit[1]*ploty + right_line.best_fit[2]

    # Measure radius of curvature in meters
    left_curverad, right_curverad = lane_curvature(ploty, left_line.bestx, right_line.bestx)
    left_line.radius_of_curvature = left_curverad
    right_line.radius_of_curvature = right_curverad

    return


def draw_lane(warped, ploty, left_fitx, right_fitx):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    return color_warp


left_line = Line()
right_line = Line()

def detect_lanes(img):
    global left_line
    global right_line

    # Apply color or gradient thresholds to the undistorted image to identify lane lines
    thresholded_binary, white_binary, yellow_binary = thresholded_binary_image(img)

    # Warp the thresholded image
    warped, Minv = warp_image(thresholded_binary)

    # Find lane lines in the warped image
    find_lines(warped, left_line, right_line)

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    color_warp = draw_lane(warped, ploty, left_line.bestx, right_line.bestx)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

    # Combine the result with the undistorted image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    offset = 30

    # Show the radius of lane curvature
    cv2.putText(result, "Radius of Curvature: {0:.1f}m".format(
                (left_line.radius_of_curvature + right_line.radius_of_curvature) / 2),
                (10, offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=2)

    # Show the vehicle position with respect to center:
    pos = vehicle_pos(warped, left_line.bestx, right_line.bestx)
    cv2.putText(result, "Vehicle is {0:.2f}m {1} of center".format(
                abs(pos), "right" if pos > 0. else "left"),
                (10, offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=2)

    return result
