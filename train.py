import pickle
import time
import os
import glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.image as mpimg
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_extraction import *

# The following are parameters for tuning
model = {}
model['color_space'] = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
model['orient'] = 9  # HOG orientations
model['pix_per_cell'] = 8 # HOG pixels per cell
model['cell_per_block'] = 2 # HOG cells per block
model['hog_channel'] = 'ALL' # Can be 0, 1, 2, or "ALL"
model['spatial_size'] = (32, 32) # Spatial binning dimensions
model['hist_bins'] = 16    # Number of histogram bins


def extract_feature_vector(filename):
    """
    Extract the features from the given image
    """

    # Read in each one by one
    img = mpimg.imread(filename)

    # The color scale of .png file mpimg read is from 0.0 to 1.0.
    # We scale it here to 0 to 255 so that the detection pipeline does not
    # have to performe scaling.
    img = (img * 255).astype(np.uint8)

    features = single_img_features(img, color_space=model['color_space'],
                        spatial_size=model['spatial_size'],
                        hist_bins=model['hist_bins'],
                        orient=model['orient'],
                        pix_per_cell=model['pix_per_cell'],
                        cell_per_block=model['cell_per_block'],
                        hog_channel=model['hog_channel'])

    return features

def extract_features(imgs):
    features = []
    with ProcessPoolExecutor() as executor:
        for feature_vector in executor.map(extract_feature_vector, imgs):
            features.append(feature_vector)

    return features

def read_data(basedir='.'):
    data = []
    image_types = os.listdir(basedir)
    for image_type in image_types:
        data.extend(glob.glob(basedir+image_type+'/*'))
    return data

def read_train_data():
    cars = read_data(basedir='data/vehicles/')
    notcars = read_data(basedir='data/non-vehicles/')

    return cars, notcars

def train():
    cars, notcars = read_train_data()
    print("# of vehicle images found: {0}".format(len(cars)))
    print("# of non-vehicle images found: {0}".format(len(notcars)))

    t = time.time()
    car_features = extract_features(cars)
    notcar_features = extract_features(notcars)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract features...')

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()

    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    return svc, X_scaler


if __name__ == '__main__':
    clf, scaler = train()
    model['clf'] = clf
    model['scaler'] = scaler
    with open('model.p', 'wb') as fout:
        pickle.dump(model, fout, protocol=pickle.HIGHEST_PROTOCOL)
