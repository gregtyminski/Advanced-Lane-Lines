import glob
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Camera():
    def __repr__(self):
        return 'Camera()'

    def __init__(self):
        # name of the file, where pickle with calibration results are stored
        self.calibration_pickle_file_name = "calibration_pickle.p"
        # path to images used to calibrate camera
        self.calibration_images = 'camera_cal/calibration*.jpg'

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane.

        # boolean indicator if camera is calibrated
        self.is_calibrated = False

        # distortion pickle
        self.dist_pickle = {}

        # calibration params
        self.mtx = None
        self.dist = None

    def calibrate(self, chess_x: int, chess_y: int, show_result: bool = False):
        '''
        `chess_x` pattern size in X direction
        `chess_y` pattern size in Y direction
        '''
        % matplotlib
        qt

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chess_y * chess_x, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chess_x, 0:chess_y].T.reshape(-1, 2)

        # Make a list of calibration images
        images = glob.glob(self.calibration_images)
        first_image = None

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (chess_x, chess_y), None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (chess_x, chess_y), corners, ret)
                if (first_image is None):
                    first_image = np.copy(img)
                # cv2.imshow('img',img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        img_size = (first_image.shape[1], first_image.shape[0])
        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size, None, None)

        % matplotlib
        inline

        if ret:
            self.is_calibrated = True
            # Save the camera calibration result for later use
            self.dist_pickle["mtx"] = mtx
            self.dist_pickle["dist"] = dist
            self.mtx = mtx
            self.dist = dist
            pickle.dump(self.dist_pickle, open(self.calibration_pickle_file_name, "wb"))

            if show_result:
                undistorted = self.undistort(first_image)
                # Visualize undistortion
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                ax1.imshow(first_image)
                ax1.set_title('Original Image', fontsize=30)
                ax2.imshow(undistorted)
                ax2.set_title('Undistorted Image', fontsize=30)

    def undistort(self, image: np.ndarray):
        '''
        `image` to be undistorted
        '''
        if not self.is_calibrated:
            return None
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)