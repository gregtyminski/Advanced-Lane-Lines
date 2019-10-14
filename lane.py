import numpy as np
from camera import Camera
from graph import Graph

class Lane():
    def __repr__(self):
        return 'Lane()'

    def __init__(self):
        # reference to Camera() obj
        self.camera = None
        #
        self.region_vertices = None

    def get_region(self, image: np.ndarray):
        if (self.region_vertices is None):
            self.region_vertices = Graph.vertices_for_region(image.shape)
        return Graph.region_of_interest(image, self.region_vertices)

    def set_camera(self, camera: Camera):
        if isinstance(camera, Camera):
            self.camera = camera

    def edge_detection(self, image: np.ndarray):
        # change to grayscale
        gray = Graph.to_grayscale(image)
        # find gradient over X
        gradx = Graph.abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(30, 100))
        # find gradient over Y
        grady = Graph.abs_sobel_thresh(gray, orient='y', sobel_kernel=3, thresh=(30, 100))
        # calculate magnitude of gradient
        mag_binary = Graph.mag_thresh(gray, sobel_kernel=9, mag_thresh=(30, 100))
        # calculate direction of gradient
        dir_binary = Graph.dir_threshold(gray, sobel_kernel=15, thresh=(0.7, 1.3))
        # change image to HLS color shape and choose 3rd channel (no 2) --> S (saturation)
        hls_binary = Graph.to_hls(image, 2, thresh=(90, 255))

        rgb_binary = Graph.color_treshold(image, 0, (200, 255))

        # combine all binary channels (gradients, magnitude, direction) and saturation
        combined = np.zeros_like(dir_binary)
        # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1
        combined[((hls_binary == 1) & (rgb_binary == 1))] = 1

        # change binary map to 3-channel color
        result = Graph.to_3channel_binary(combined)
        return result

    def pipeline(self, image: np.ndarray):
        # undistort image
        undistorted = self.camera.undistort(image)

        region = Graph.region_of_interest(undistorted, Graph.vertices_for_region(undistorted.shape))
        mean_lightness = Graph.mean_lightness(region)
        # print(mean_lightness)

        unbrightned = Graph.adjust_brightness(undistorted)
        # detect edges
        combined = lane.edge_detection(unbrightned)

        # change perspective
        warped, M = Graph.get_perspective_transform(combined)

        # src = Graph.vertices_for_region(image.shape)
        # dst = Graph.destination_vertices(image.shape)

        # print(src, dst)
        # Graph.draw_lines(image, src[0], color = (0, 255, 0), thickness = 4)
        # Graph.draw_lines(image, dst[0], color = (255, 0, 0), thickness = 4)

        # pick 1 channel
        warped_1channel = warped[:, :, 0]
        # calculate histogram
        # hist = Graph.histogram(warped_1channel)

        fitted = Graph.fit_polynomial(warped_1channel)
        # leftx, lefty, rightx, righty, out_img = Graph.find_lane_pixels(warped_1channel)

        result = fitted
        # text = 'Bright = ' + str(int(mean_lightness))
        # cv2.putText(result, text, (50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 3)
        return result