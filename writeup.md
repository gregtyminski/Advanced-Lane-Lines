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

### 4. Find lane pixels

#### a) histogram

#### b) divide line onto N parts

#### c) find center points

#### d) calculate distance between lines

### 5. Fit both lines

### 6. Warp perspective (of undistorted image) to bird eye view

### 7. Draw lines and lane

### 8. Reverse perspective warp to camera view

### 9. Draw reversed image with lines & lane on original undistorted image

### 10. Calculate curvature and vehicle position

### 11. Print radius of curvature and position on the image


---

## Discussion

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---
