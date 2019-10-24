import numpy as np
import matplotlib.pyplot as plt
from camera import Camera
from graph import Graph
import cv2

class Lane():
    def __repr__(self):
        return 'Lane()'

    def __init__(self, lane_width_m: float, lane_length_m: float):
        # reference to Camera() obj
        self.camera = None
        #
        self.region_vertices = None
        # distance in meters per pixel in Y and X directions
        self.m_per_ypx = None
        self.m_per_xpx = None
        # distance in pixels between lines
        self.lane_width_px = []
        # distance in meters between lines
        self.lane_width_m = lane_width_m
        # visible lane distance in meters
        self.lane_length_m = lane_length_m
        # visible lane distance in px
        self.lane_length_px = None
        # history of curvature
        self.curvature_hist = []
        # curvature constant
        self.curvature_const = 9
        # bottom line index
        self.bottom_line_index = None
        # last left x base
        self.last_left_base = None
        # last right x base
        self.last_right_base = None

    def get_region(self, image: np.ndarray):
        if self.region_vertices is None:
            self.region_vertices = Graph.vertices_for_region(image.shape)
        return Graph.region_of_interest(image, self.region_vertices)

    def set_camera(self, camera: Camera):
        if isinstance(camera, Camera):
            self.camera = camera

    def edge_detection(self, image: np.ndarray):
        # change to grayscale
        # gray = Graph.to_grayscale(image)
        imgcopy = image.copy()

        # Convert to HLS color
        hls = np.float32(cv2.cvtColor(imgcopy, cv2.COLOR_RGB2HLS))
        s_channel = Graph.to_hls(imgcopy, 2, thresh=(90, 255))
        r_channel = Graph.color_treshold(imgcopy, 0, (200, 255))
        channel = np.zeros_like(s_channel)
        channel[((s_channel == 1) & (r_channel == 1) )] = 1
        # Sobel x
        # sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0)  # Take the derivative in x

        # Threshold x gradient
        #sxbinary = np.zeros_like(sobelx)
        #sxbinary[(sobelx > 10) & (sobelx <= 255)] = 255.

        # Threshold color channel
        #s_binary = np.zeros_like(s_channel)
        #s_binary[(s_channel > 100) & (s_channel <= 255)] = 255.
        #plt.imshow(s_channel, cmap='gray')
        #plt.show()
        #plt.imshow(r_channel, cmap='gray')
        #plt.show()
        #plt.imshow(channel, cmap='gray')
        #print(np.max(s_channel), np.max(r_channel), np.max(channel))

        #gradx = Graph.abs_sobel_thresh(channel, orient='x', sobel_kernel=3, thresh=(30, 100))
        #plt.imshow(gradx, cmap='gray')
        #plt.show()
        # find gradient over X
        # gradx = Graph.abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(30, 100))
        # find gradient over Y
        # grady = Graph.abs_sobel_thresh(gray, orient='y', sobel_kernel=3, thresh=(30, 100))
        # calculate magnitude of gradient
        # mag_binary = Graph.mag_thresh(gray, sobel_kernel=9, mag_thresh=(30, 100))
        # calculate direction of gradient
        # dir_binary = Graph.dir_threshold(gray, sobel_kernel=15, thresh=(0.7, 1.3))
        # change image to HLS color shape and choose 3rd channel (no 2) --> S (saturation)
        # hls_binary = Graph.to_hls(image, 2, thresh=(90, 255))

        # rgb_binary = Graph.color_treshold(image, 0, (190, 255))

        # combine all binary channels (gradients, magnitude, direction) and saturation
        # combined = np.zeros_like(hls_binary)
        # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1
        # combined[((hls_binary == 1) & (rgb_binary == 1) )] = 1

        # change binary map to 3-channel color
        result = Graph.to_3channel_binary(channel)
        return result

    def pipeline(self, image: np.ndarray):
        # undistort image
        undistorted = self.camera.undistort(image)

        # brightned = Graph.adjust_brightness(undistorted)
        # detect edges
        combined = self.edge_detection(undistorted)

        # change perspective
        warped_edges, M = Graph.get_perspective_transform(combined)
        warped_frame, M = Graph.get_perspective_transform(undistorted)

        # pick 1 channel
        warped_1channel = warped_edges[:, :, 0]
        # calculate histogram
        # hist = Graph.histogram(warped_1channel)
        # src = Graph.vertices(image.shape)
        # dst = Graph.destination_vertices(image.shape)
        # Graph.draw_lines(undistorted, src[0], color = (0, 255, 0), thickness = 4)
        # Graph.draw_lines(undistorted, dst[0], color = (255, 0, 0), thickness = 4)

        fitted, left_fit, left_fitx, right_fit, right_fitx, ploty, avg_lane_dist, leftx_base, rightx_base = Graph.fit_polynomial(warped_1channel, self.last_left_base, self.last_right_base)
        self.last_left_base = leftx_base
        self.last_right_base = rightx_base
        # update distances (width and length) of lane
        self.__update_distances(avg_lane_dist, fitted.shape[0])

        lane_drawn = Graph.draw_lanes(warped_frame, left_fitx, right_fitx)
        lane_reversed, M = Graph.get_perspective_transform(lane_drawn, reverse=True)

        # create mask for merging original image with drawn lane image got from reverse perspective transform
        mask = np.expand_dims(
            ((lane_reversed[:, :, 0] == 0) & (lane_reversed[:, :, 1] == 0) & (lane_reversed[:, :, 2] == 0)), axis=2)
        lane_reversed = mask * undistorted + (1 - mask) * lane_reversed
        lane_reversed = np.array(lane_reversed, dtype=np.uint8)

        # position of vehicle on lane
        lane_pos = self.__find_position_on_lane(int(warped_1channel.shape[1]//2), leftx_base, rightx_base)
        direction_text = ' to left'
        if lane_pos < 0:
            direction_text = ' to right'
        elif lane_pos == 0:
            direction_text = ' '
        # curvature (50 last calculations are kept and average value is returned)
        if self.bottom_line_index is None:
            self.bottom_line_index = np.max(ploty)
        curvature = self.__find_curvature(left_fit, self.bottom_line_index)
        curvature = self.__find_curvature(right_fit, self.bottom_line_index)

        # print texts on the frame with radius of curvature and distance from midpoint
        result = lane_reversed
        if curvature > 3000.:
            dist_text = 'Radius of curvature = straight lane'
        else:
            dist_text = 'Radius of curvature = ' + str("%.2f" % curvature) + ' meters'
        cv2.putText(result, dist_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
        dist_text = 'Distance from mid lane = ' + str("%.2f" % lane_pos) + ' meters' + direction_text
        cv2.putText(result, dist_text, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
        return result

    def __update_distances(self, lane_width_px: int, lane_length_px: int):
        self.lane_width_px.append(lane_width_px)
        self.m_per_xpx = self.lane_width_m / np.average(self.lane_width_px)
        if self.lane_length_px is None:
            self.lane_length_px = lane_length_px
            self.m_per_ypx = self.curvature_const * self.lane_length_m / self.lane_length_px

    def __find_curvature(self, line_fit_coeffs: np.ndarray, y: int):
        poly = np.polynomial.polynomial.Polynomial(line_fit_coeffs[::-1])

        # first derivative
        d1 = poly.deriv()
        # second derivative
        d2 = d1.deriv()
        # curvature
        result = ((1 + d1(y)**2)**1.5) / np.abs(d2(y))

        self.curvature_hist.append(result)

        if len(self.curvature_hist) > 50:
            self.curvature_hist.pop(0)
        return np.average(self.curvature_hist) * self.m_per_ypx

    def __find_position_on_lane(self, midpoint: int, leftx_base: int, rightx_base: int):
        '''
        Find distance from midpoint of the lane.
        Values for params are returned by Graph.find_lane_pixels
        :param midpoint: Midpoint of warped lane image.
        :param leftx_base: Base point of left line.
        :param rightx_base: Base point of right line.
        :return: Distance in meters from midpoint. If negative, vehicle moved to left. If positive, vehicle moved to right.
        '''
        return ((rightx_base + leftx_base)/2 - midpoint) * self.m_per_xpx
