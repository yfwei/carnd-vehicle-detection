import argparse
import time
import pickle
from concurrent.futures import ProcessPoolExecutor
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label
from feature_extraction import *
from lane_detect import *


# Read the model parameters from the pickle file
with open('model.p', 'rb') as fin:
    model = pickle.load(fin)
color_space = model['color_space']
orient = model['orient']
pix_per_cell = model['pix_per_cell']
cell_per_block = model['cell_per_block']
hog_channel = model['hog_channel']
spatial_size = model['spatial_size']
hist_bins = model['hist_bins']
clf = model['clf']
scaler = model['scaler']

# Min and max in y to search in slide_window()
y_start_stop = [400, 656]

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def find_cars(cars, labels):
    cars_detected = []

    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        car = Vehicle()
        # Define a bounding box based on min/max x and y
        car.bbox = np.asarray([np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)])

        cars_detected.append(car)

    if len(cars) > 0:
        # Detect new cars
        new_cars = []
        for car_detected in cars_detected:
            overlapped = False
            for car in cars:
                if car.overlapped(car_detected):
                    overlapped = True
                    car.add_bbox(car_detected.bbox)

            if not overlapped:
                car_detected.n_detections = 1
                new_cars.append(car_detected)

        # Remove cars that are not detected in 60 consesutive frames
        for car in cars:
            car.detected = False
            for car_detected in cars_detected:
                if car.overlapped(car_detected):
                    car.detected = True
                    car.n_detections += 1
                    car.n_nondetections = 0
            if not car.detected:
                car.n_nondetections += 1
        for car in cars:
            if car.n_nondetections >= 30:
                cars.remove(car)

        cars += new_cars
    else:
        cars += cars_detected

    return cars

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0):
    on_windows = []

    for window in windows:
        #1) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #2) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel)
        #3) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #4) Predict using your classifier
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
    return on_windows


class Vehicle():
    def __init__(self):
        self.detected = False

        # The average bounding box over last few frames
        self.bbox = None

        # The last n bounding boxes
        self.recent_bboxes = []

        # Number of times this vehicle has been detected.
        self.n_detections = 0

        # Number of times this vehicle has not been detected.
        self.n_nondetections = 0

        self.weights = np.linspace(0.0, 1.0, num=15 + 1)
        self.weights = self.weights[1:]

    def add_bbox(self, bbox):
        self.recent_bboxes.append(bbox)
        if len(self.recent_bboxes) > 15:
            self.recent_bboxes.pop(0)

        weights = self.weights[:len(self.recent_bboxes)]
        weights /= np.sum(weights)

        self.bbox = np.average(
                        np.asarray(self.recent_bboxes),
                        weights=weights,
                        axis=0).astype(np.uint32)

    def overlapped(self, car):
        dx = min(car.bbox[2], self.bbox[2]) - max(car.bbox[0], self.bbox[0])
        dy = min(car.bbox[3], self.bbox[3]) - max(car.bbox[1], self.bbox[1])
        return (dx >= 0) and (dy >= 0)


cars = []
heatmaps = []

def process_image(img):
    global cars
    global heatmaps

    windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
                    xy_window=(96, 96), xy_overlap=(0.75, 0.75))
    windows += slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop,
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    hot_windows = []

    # Split the search ranges to utilize the multiple processors
    windows_list = []
    offset = 20
    n_list = len(windows) // offset
    for i in range(n_list):
        windows_list.append(windows[i * offset:(i + 1) * offset])

    futures = []
    with ProcessPoolExecutor() as executor:
        for windows in windows_list:
            future = executor.submit(search_windows, img, windows, clf, scaler, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel)
            futures.append(future)
        for future in futures:
            windows = future.result()
            hot_windows += windows

    heatmap = np.zeros_like(img[:,:,0])
    heatmap = add_heat(heatmap, hot_windows)

    # Keep the last 8 heatmaps
    heatmaps.append(heatmap)
    if len(heatmaps) > 8:
        heatmaps.pop(0)
    heatmap = np.sum(np.asarray(heatmaps), axis=0)

    # Apply threshold to reject false positives
    heatmap = apply_threshold(heatmap, 12)

    labels = label(heatmap)
    cars = find_cars(cars, labels)

    img = detect_lanes(img)

    # Draw bounding boxes around the detected cars
    for car in cars:
        cv2.rectangle(img, (car.bbox[0], car.bbox[1]),
                  (car.bbox[2], car.bbox[3]), (0,0,255), 6)

    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vehicle Detector')
    parser.add_argument(
        'video',
        type=str,
        default='',
        nargs='?',
        help='Path to the video file.'
    )

    args = parser.parse_args()

    if args.video != '':
        video_output = 'output_videos/' + args.video

        clip1 = VideoFileClip(args.video)
        white_clip = clip1.fl_image(process_image)
        white_clip.write_videofile(video_output, audio=False)
