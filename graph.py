import cv2
import numpy as np
import matplotlib.pyplot as plt


class Graph():
    # region of interest coefficients
    # Polygon has shape (I'll correct numbers in comment at the end):
    # ```````````````````````````````````````````
    # `                 (0.593)                 `
    # `      (0.47)_________________(0.52)      `
    # `           |                 \           `
    # `          |                   \          `
    # `         |                     \         `
    # ` (0.0) ------------------------- (1.0)  `
    # `                 (1.0)                  `
    # ```````````````````````````````````````````
    LEFT_X_BOTTOM_COEF = 0.1428571429
    RIGHT_X_BOTTOM_COEF = 1.0 - LEFT_X_BOTTOM_COEF
    LEFT_X_TOP_COEF = 0.45
    RIGHT_X_TOP_COEF = 1.0 - LEFT_X_TOP_COEF
    Y_TOP_COEF = 0.642857142857143
    Y_BOTTOM_COEF = 1.0

    SOURCE_COEFFS = (LEFT_X_BOTTOM_COEF, RIGHT_X_BOTTOM_COEF,
                     LEFT_X_TOP_COEF, RIGHT_X_TOP_COEF,
                     Y_TOP_COEF, Y_BOTTOM_COEF)

    DESTINATION_COEFFS = (0.166666666666667, 0.833333333333333, LEFT_X_TOP_COEF, RIGHT_X_TOP_COEF,
                          Y_TOP_COEF, Y_BOTTOM_COEF)

    CLAHE = None

    @staticmethod
    def to_grayscale(image: np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def to_3channel_binary(image: np.ndarray):
        int_image = image.astype(dtype=np.uint8)
        color_binary = np.dstack((int_image, int_image, int_image)) * 255
        return color_binary

    @staticmethod
    def to_1channel_binary(image: np.ndarray):
        int_image = image.astype(dtype=np.uint8)
        color_binary = np.dstack((int_image)) * 255
        return color_binary

    @staticmethod
    def histogram_gray(image: np.ndarray):
        '''
        `image` grascaled image

        Returns histogram of this image.
        '''
        # Grab only the bottom half of the image
        # Lane lines are likely to be mostly vertical nearest to the car
        sizey, sizex = image.shape
        bottom_half = image[sizey // 2:, :]

        # Sum across image pixels vertically - make sure to set an `axis`
        # i.e. the highest areas of vertical lines should be larger values
        histogram = np.sum(bottom_half, axis=0)

        return histogram

    # Define a function that applies Sobel x or y,
    # then takes an absolute value and applies a threshold.
    # Note: calling your function with orient='x', thresh_min=20, thresh_max=100
    # should produce output like the example image shown above this quiz.
    @staticmethod
    def abs_sobel_thresh(image: np.ndarray, orient: str = 'x', sobel_kernel: int = 3, thresh=(0, 255)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        else:
            return np.copy(image)

        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)

        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # 5) Create a mask of 1's where the scaled gradient magnitude
        # is > thresh_min and < thresh_max
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        return sxbinary

    # Define a function that applies Sobel x and y,
    # then computes the magnitude of the gradient
    # and applies a threshold
    @staticmethod
    def mag_thresh(image: np.ndarray, sobel_kernel: int = 3, mag_thresh=(0, 255)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Calculate the magnitude
        abs_sobel = np.absolute(np.sqrt(np.power(sobelx, 2) + np.power(sobely, 2)))

        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        # 5) Create a binary mask where mag thresholds are met
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        return binary_output

    # Define a function that applies Sobel x and y,
    # then computes the direction of the gradient
    # and applies a threshold.
    @staticmethod
    def dir_threshold(image: np.ndarray, sobel_kernel: int = 3, thresh=(0, np.pi / 2)):
        '''
        `image` image in grayscale to be processed
        `sobel_kernel`
        `tresh` is an array where [0] is lower treshold and [1] is upper treshold
        '''
        # Apply the following steps to img
        # 1) Convert to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # 3) Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)

        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        scaled_sobel = np.arctan2(abs_sobely, abs_sobelx)

        # 5) Create a binary mask where direction thresholds are met
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # 6) Return this mask as your binary_output image
        return binary_output

    def color_treshold(image: np.ndarray, channel: int, tresh: tuple = (200, 255)):
        rgb_binary = np.zeros_like(image[:, :, channel])
        rgb_binary[(image[:, :, channel] >= tresh[0]) & (image[:, :, channel] <= tresh[1])] = 1

        return rgb_binary

    @staticmethod
    def to_hls(image: np.ndarray, color_numb: int = 2, thresh=(0, 255)):
        # 1) Convert to HLS color space
        hls_color = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        hls = hls_color[:, :, color_numb]
        # 2) Apply a threshold to the S channel
        binary_output = np.zeros_like(hls)
        binary_output[(hls >= thresh[0]) & (hls <= thresh[1])] = 1

        # 3) Return a binary image of threshold result
        return binary_output

    @staticmethod
    def to_rgb(image: np.ndarray, color_numb: int = 0, thresh=(0, 255)):
        rgb = image[:, :, color_numb]
        binary_output = np.zeros_like(rgb)
        binary_output[(rgb >= thresh[0]) & (rgb <= thresh[1])] = 1
        return binary_output

    @staticmethod
    def mean_lightness(image: np.ndarray):
        # 1) Convert to HLS color space
        hls_color = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        lightness = hls_color[:, :, 1]
        # 2) Apply a threshold to the S channel
        return np.mean(lightness)

    @staticmethod
    def region_of_interest(image: np.ndarray, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        copy = np.copy(image)
        # defining a blank mask to start with
        mask = np.zeros_like(copy)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(image.shape) > 2:
            channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices.astype(int), ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(copy, mask)
        return masked_image

    @staticmethod
    def draw_lines(image: np.ndarray, points: tuple, color: tuple = (255, 0, 0), thickness: int = 4):
        point1 = points[0]
        point2 = points[1]
        point3 = points[2]
        point4 = points[3]
        cv2.line(image, (point1[0], point1[1]), (point2[0], point2[1]), color, thickness)
        cv2.line(image, (point2[0], point2[1]), (point3[0], point3[1]), color, thickness)
        cv2.line(image, (point3[0], point3[1]), (point4[0], point4[1]), color, thickness)
        cv2.line(image, (point4[0], point4[1]), (point1[0], point1[1]), color, thickness)

    @staticmethod
    def get_perspective_transform(image: np.ndarray, reverse: bool = False):
        image_shape = image.shape
        src = Graph.vertices_for_region(image_shape, Graph.SOURCE_COEFFS)
        dst = Graph.destination_vertices(image_shape, Graph.DESTINATION_COEFFS)
        if not reverse:
            M = cv2.getPerspectiveTransform(src, dst)
        else:
            M = cv2.getPerspectiveTransform(dst, src)
        xs, ys = image.shape[1], image.shape[0]
        warped = cv2.warpPerspective(image, M, (xs, ys), flags=cv2.INTER_LINEAR)
        return warped, M

    @staticmethod
    def destination_vertices(image_shape: tuple, coefficients: tuple = None):
        coefficients = Graph.DESTINATION_COEFFS
        ysize, xsize = image_shape[0], image_shape[1]

        left_coef = coefficients[0]
        right_coef = coefficients[1]
        left_top_coef = coefficients[2]
        right_top_coef = coefficients[3]
        up_line_coef = coefficients[4]
        bottom_line_coef = coefficients[5]
        point1 = (int(right_coef * xsize), 0)
        point2 = (int(right_coef * xsize), int(bottom_line_coef * ysize))
        point3 = (int(left_coef * xsize), int(bottom_line_coef * ysize))
        point4 = (int(left_coef * xsize), 0)

        vertices = np.array([[point1, point2, point3, point4]], dtype=np.float32)
        return vertices

    @staticmethod
    def vertices_for_region(image_shape: tuple, coefficients: tuple = None):
        coefficients = Graph.SOURCE_COEFFS
        ysize, xsize = image_shape[0], image_shape[1]

        left_coef = coefficients[0]
        right_coef = coefficients[1]
        left_top_coef = coefficients[2]
        right_top_coef = coefficients[3]
        up_line_coef = coefficients[4]
        bottom_line_coef = coefficients[5]
        point1 = (int(right_top_coef * xsize), int(up_line_coef * ysize))
        point2 = (int(right_coef * xsize), int(bottom_line_coef * ysize))
        point3 = (int(left_coef * xsize), int(bottom_line_coef * ysize))
        point4 = (int(left_top_coef * xsize), int(up_line_coef * ysize))
        vertices = np.array([[point1, point2, point3, point4]], dtype=np.float32)
        return vertices

    @staticmethod
    def histogram(image: np.ndarray):
        # 1 channel only
        # Grab only the bottom half of the image
        # Lane lines are likely to be mostly vertical nearest to the car
        sizey, sizex = image.shape[0], image.shape[1]
        bottom_half = image[sizey // 2:, :]

        # Sum across image pixels vertically - make sure to set an `axis`
        # i.e. the highest areas of vertical lines should be larger values
        histogram = np.sum(bottom_half, axis=0)

        return histogram

    def adjust_brightness(image: np.ndarray):
        hls_color = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        # light_binary = Graph.to_hls
        # mean = Graph.mean_lightness(image)

        # sizey, sizex = hls_color.shape[0], hls_color.shape[1]
        light_channel = np.array(hls_color[:, :, 1], dtype=np.uint8)
        sat_channel = np.array(hls_color[:, :, 2], dtype=np.uint8)
        hue_channel = np.array(hls_color[:, :, 0], dtype=np.uint8)

        # create a CLAHE object (Arguments are optional).
        if (Graph.CLAHE is None):
            Graph.CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        light_channel = Graph.CLAHE.apply(light_channel)
        # sat_channel = Graph.CLAHE.apply(sat_channel)
        # hue_channel = Graph.CLAHE.apply(hue_channel)

        stacked = np.dstack((hue_channel, light_channel, sat_channel))
        result = cv2.cvtColor(stacked, cv2.COLOR_HLS2RGB)

        return result

    @staticmethod
    def merge_images(images):
        return np.concatenate(images, axis=1)

    @staticmethod
    def fit_polynomial(binary_warped):
        # Find our lane pixels first
        left_points, right_points, out_img, avg_lane_dist, midpoint, leftx_base, rightx_base = Graph.find_lane_pixels(binary_warped)

        # fit 2nd order left line
        if len(left_points)>=3:
            left_fit = np.polyfit(left_points[:,1], left_points[:,0], deg=2)
        # fit 2nd order right line
        if len(right_points)>=3:
            right_fit = np.polyfit(right_points[:,1], right_points[:,0], deg=2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        try:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        return out_img, left_fit, left_fitx, right_fit, right_fitx, ploty, avg_lane_dist, midpoint, leftx_base, rightx_base

    @staticmethod
    def find_lane_pixels(binary_warped: np.ndarray):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        # initial position of left line on warped image
        leftx_base = np.argmax(histogram[:midpoint])
        # initial position of right line on warped image
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 15
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # we need to track previous distance between lines to be able to approximate the mid points and right track of lines
        prev_lane_dist = rightx_base - leftx_base
        # we need to track all distances between lines to calculate average distance
        lane_dist_hist = [prev_lane_dist]

        yellow_color = (255, 255, 153)

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_points = []
        right_lane_points = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # find center of the 'lane region'
            left_area = binary_warped[win_y_low:win_y_high, win_xleft_low:win_xleft_high]
            right_area = binary_warped[win_y_low:win_y_high, win_xright_low:win_xright_high]
            l_area_hist = np.sum(left_area, axis=0)
            r_area_hist = np.sum(right_area, axis=0)
            l_x_index = np.argmax(l_area_hist) + win_xleft_low
            r_x_index = np.argmax(r_area_hist) + win_xright_low
            y_index = int((win_y_low + win_y_high) / 2)

            # didn't find center
            if l_x_index == win_xleft_low:
                l_x_index = None
            if r_x_index == win_xright_low:
                r_x_index = None
            # finding distance and keep it
            if (l_x_index is not None) and (r_x_index is not None):
                prev_lane_dist = (r_x_index - l_x_index)
                lane_dist_hist.append(prev_lane_dist)
            # adjusting lane
            elif (l_x_index is None) and (r_x_index is None):
                l_x_index = int((win_xleft_low + win_xleft_high) / 2)  # guessing value
                r_x_index = int((win_xright_low + win_xright_high) / 2)  # guessing value
            elif (l_x_index is None) and (r_x_index is not None):
                l_x_index = r_x_index - prev_lane_dist
            elif (l_x_index is not None) and (r_x_index is None):
                r_x_index = l_x_index + prev_lane_dist

            l_point = (l_x_index, y_index)
            r_point = (r_x_index, y_index)

            left_lane_points.append(l_point)
            right_lane_points.append(r_point)

            leftx_current = l_x_index
            rightx_current = r_x_index

            ## Visualization ##
            # Colors in the left and right lane regions
            # Draw the windows on the visualization image
            # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
            # yellow center points
            cv2.circle(out_img, l_point, 2, yellow_color, thickness=2, lineType=8)
            cv2.circle(out_img, r_point, 2, yellow_color, thickness=2, lineType=8)

        avg_lane_dist = np.average(lane_dist_hist)
        return np.array(left_lane_points), np.array(right_lane_points), out_img, avg_lane_dist, midpoint, leftx_base, rightx_base

    @staticmethod
    def draw_lanes(binary_warped: np.ndarray, left_fitx: np.ndarray, right_fitx: np.ndarray):
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        red_color = [255, 0, 0]
        blue_color = [0,0, 255]
        green_color = [0, 255, 0]
        thickness = 5

        l_points = np.vstack((left_fitx, ploty)).T
        r_points = np.vstack((right_fitx, ploty)).T

        # draw both lines
        pts = np.array(l_points, np.int32)
        cv2.polylines(binary_warped, [pts], False, red_color, thickness)
        pts = np.array(r_points, np.int32)
        cv2.polylines(binary_warped, [pts], False, blue_color, thickness)

        # fill area between
        all_points = np.vstack((l_points, np.flipud(r_points)))
        pts = np.array(all_points, np.int32)
        #print(pts.shape)
        cv2.fillConvexPoly(binary_warped, pts, green_color)
        return binary_warped