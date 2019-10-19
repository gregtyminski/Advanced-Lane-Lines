# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[calibration]: ./output_images/calibration.png "Calibration"
[undistorted]: ./output_images/undistorted_test3.jpg "Undistorted image"
[combined]: ./output_images/combined_test3.jpg "Edges detected"
[src_vertices]: ./output_images/src_vertices_test3.jpg "Source region"
[warped]: ./output_images/warped_edges_test3.jpg "Warped perspective of binary image"
[slices]: ./output_images/fitted_test3.jpg "Central points found"
[warpedlane]: ./output_images/warped_lane_test3.jpg "Warped lane"
[lanedrawn]: ./output_images/warped_frame_test3.jpg "Lines and lane drawn"
[alldrawn]: ./output_images/lane_reversed_test3.jpg "Perspective reversed"
[result]: ./output_images/result_test3.jpg "Result image"

## Before Pipeline

### Camera calibration
Before conducting any lane finding we need to ensure, that the image is not distorted.
To do so calibration of camera is necessary.
For this purpose `Camera` class is defined in the file `camera.py`.

To calibrate camera just use following code:
```python
# create camera object
camera = Camera()
# calibrate camera and show result
camera.calibrate(9, 6, True)
```
... where `9` and `6` are the number of checks in columns and rows on source images.
If you don't specify source folder with the images for calibration the default one is taken: `camera_cal/calibration*.jpg`

To change source folder use following code before running calibration:
```python
camera.__set_source_images('camera_cal/calibration*.jpg')
camera.calibrate(9, 6, True)

```
When you specify `show_result=True` while calling calibration method, the result will be shown:
![alt text][calibration]

When camera is calibrated ... it's ready for undistorting images.

## Pipeline

Finding lane is implemented in the `Lane` class defined in the `lane.py` file.
For all graphics operations a `Graph` class is defined with static methods doing operations on images.
`Graph` class is defined in `graph.py` file.

For **each image** please create **new Lane()** object to clear the history.
Lane class keeps history of previous frames.
```python
lane = Lane(lane_width_m=3.7, lane_length_m=30.)
lane.set_camera(camera)
```
While creating instance of `Lane` object, specify width of lane in meters and width of lane (shall be 30m).

If you are running pipeline on a video, where each next frame relates to the same street, **keep 1 `Lane` object**.
When you have `Lane` object, you can use it for lane detection.

All the magic happens just by calling `pipeline` method:
```python
result = lane.pipeline(image)
```
Where `result` is an image after entire pipeline.


**Let's see, what happens inside `pipeline` method:**

### 1. Undistort the image

First step is to undistort the image (camera is calibrated):
```python
# undistort image
undistorted = self.camera.undistort(image)
```
In a result we get:
![alt text][undistorted]

### 2. Detect edges
Next step is to find edges in an image:
```python
# detect edges
combined = self.edge_detection(undistorted)
```

The edges detection takes following steps:\
* calculate grayscaled image `gray = Graph.to_grayscale(image)`\
* calculate binary image from tresholded __saturation__ channel of original image `hls_binary = Graph.to_hls(image, 2, thresh=(90, 255))`
* calculate binary image from tresholded __red__ channel of original image `rgb_binary = Graph.color_treshold(image, 0, (200, 255))`
* combine `hls_binary` and `rgb_binary` into one binary image `combined[((hls_binary == 1) & (rgb_binary == 1))] = 1`

In a result following edges are found:
![alt text][combined]

### 3. Warp perspective (of binary warped image) to bird eye view
When the edges are found, we need to warp central part of the image to bird eye view.
Source region for warping an image is defined with following coefficients of an image, which are defined in `Graph` class in `grap.py` file
```python
    # ```````````````````````````````````````````
    # `                 (0.643)                 `
    # `      (0.45)_________________(0.55)      `
    # `           |                 \           `
    # `          |                   \          `
    # `         |                     \         `
    # ` (0.143) ---------------------- (0.857)  `
    # `                 (1.0)                   `
    # ```````````````````````````````````````````
```
which corresponds to following region on the screen:
![alt text][src_vertices]

To warp perspective following code is used:
```python
warped_edges, M = Graph.get_perspective_transform(combined)
```
and the following result is returned:
![alt text][warped]

We need to ensure, that we are working on 1-channel only.
`warped_1channel = warped_edges[:, :, 0]`

### 4. Find lane pixels
All the magic in _Find lane pixels_ happens inside the method `Graph.find_lane_pixels(binary_warped)`

#### a) histogram
When we have perspective warped to bird eye view, we need calculate histogram over y-axis.
`histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)`

Within the histogram we need to detect position of both lanes
```python
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0] // 2)
# initial position of left line on warped image
leftx_base = np.argmax(histogram[:midpoint])
# initial position of right line on warped image
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```
#### b) divide line onto N parts
Next step is to divide each detected line onto N parts (I use 15).

#### c) find center points
In each little part we find central point.
How? Just by calculating histogram of the little part only and find the peak value of it.
```python
# find center of the 'lane region'
left_area = binary_warped[win_y_low:win_y_high, win_xleft_low:win_xleft_high]
right_area = binary_warped[win_y_low:win_y_high, win_xright_low:win_xright_high]
l_area_hist = np.sum(left_area, axis=0)
r_area_hist = np.sum(right_area, axis=0)
l_x_index = np.argmax(l_area_hist) + win_xleft_low
r_x_index = np.argmax(r_area_hist) + win_xright_low
y_index = int((win_y_low + win_y_high) / 2)
```
Those central points wiil be used to fit lines.
As a result we get following picture.
![alt text][slices]

#### d) calculate distance between lines
For more precise calculation of central points, distance of lines for each little part and kept in list with history of them:
```python
prev_lane_dist = (r_x_index - l_x_index)
lane_dist_hist.append(prev_lane_dist)
```

### 5. Fit both lines
When we have set of central points for each of the line, we can fit them to polynomila describing both lines.
```python
# fit 2nd order left line
if len(left_points)>=3:
    left_fit = np.polyfit(left_points[:,1], left_points[:,0], deg=2)
# fit 2nd order right line
if len(right_points)>=3:
    right_fit = np.polyfit(right_points[:,1], right_points[:,0], deg=2)
```

### 6. Warp perspective (of undistorted image) to bird eye view
When all lines are detected and polynomials calculated, we need to warp perspective to bird eye view from original image in the same way as the binary image previously.
```python
warped_frame, M = Graph.get_perspective_transform(undistorted)
```
``![alt text][warpedlane]`

### 7. Draw lines and lane
On this newly warped original image we need to draw both lines and lane itself `lane_drawn = Graph.draw_lanes(warped_frame, left_fitx, right_fitx).
```python
# draw both lines
pts = np.array(l_points, np.int32)
cv2.polylines(warped_frame, [pts], False, red_color, thickness)
pts = np.array(r_points, np.int32)
cv2.polylines(warped_frame, [pts], False, blue_color, thickness)

# fill area between
all_points = np.vstack((l_points, np.flipud(r_points)))
pts = np.array(all_points, np.int32)
#print(pts.shape)
cv2.fillConvexPoly(warped_frame, pts, green_color)
````
``![alt text][lanedrawn]`

### 8. Reverse perspective warp to camera view and draw it back on original image
Next step is to warp perspective back from bird eye view to driver view.
```python
lane_reversed, M = Graph.get_perspective_transform(lane_drawn, reverse=True)
```
... and then draw it back on original undistorted image.
It's done by creating a mask and copying un-warped image in masked area.
```python
# create mask for merging original image with drawn lane image got from reverse perspective transform
mask = np.expand_dims(((lane_reversed[:,:,0] == 0) & (lane_reversed[:,:,1]==0) & (lane_reversed[:,:,2]==0)), axis=2)
lane_reversed = mask * undistorted + (1 - mask) * lane_reversed
lane_reversed = np.array(lane_reversed, dtype=np.uint8)
```
``![alt text][alldrawn]`

### 10. Calculate curvature and vehicle position
Last step is to calculate position of vehicle on the line and radius of curvature of the lane.
Position on the lane is calculated in the method:
```python
lane_pos = self.__find_position_on_lane(midpoint, leftx_base, rightx_base)
```

Finding radius of curvature is a bit more complicated.
For each line on each frame radius is calculated using this [formula](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) which is implemented in `l_curvature = self.__find_curvature(left_fit)`
20 calculated radius values are kept in history of radiuses and average value is returned.

### 11. Print radius of curvature and position on the image
The very last step is just to print text on a frame with calculated values.
```python
dist_text = 'Radius of curvature = ' + str("%.2f" % curvature) + ' meters'
cv2.putText(result, dist_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)
dist_text = 'Distance from mid lane = ' + str("%.2f" % lane_pos) + ' meters' + direction_text
cv2.putText(result, dist_text, (50,100), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 3)
```
``![alt text][result]`

## Result on video

The result of processing the pipeline can be visible on the processing original video --> [project_video.mp4](project_video.mp4)\
The result is here --> [result_project_video](output_images\project_video.mp4)

---

## Discussion
The entire pipeline needs few improvements.
* I have tried several options to tweak hyperparams and usage different `Sobel x & y` as well as `magnitude` edge detections. For the time being current implementation is just OK. But it does not suit all conditions like shadow or over-lighted areas of lane like in `challenge` and `harder_challenge` videos. I've implemented method (included in source code) for correcting histogram of lightness channel in HLS color map. --> `Graph.adjust_brightness(image)` This could be used to correct lightness and improve the pipeline. I didn't manage to finish it due to lack of time.
* Another improvement would be to run hyperparams optimization (check range of values) and verify which are best in certain lightining conditions and on each frame apply different hyperparams based on lightning.
* Entire pipeline needs performance improvements and usage of _faster_ calculation methods. This will be very crucial when installed in a real car.

---
