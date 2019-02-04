# **Advanced Lane Finding Project**

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### 1. PIPELINE DESCRIPTION
The Software Pipeline built for Advanced Lane Detection consists of the following steps:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

### Camera Calibration and Distortion Correction
### Code Snapshots
![camera_calib_codesnapshot_](https://user-images.githubusercontent.com/25223180/52207620-fb281880-28a3-11e9-87b1-9dd46d3a887f.PNG)
![camera_calib_codesnapshot_2](https://user-images.githubusercontent.com/25223180/52207625-ff543600-28a3-11e9-8e37-85b084195797.PNG)
![camera_calib_codesnapshot_3](https://user-images.githubusercontent.com/25223180/52207629-024f2680-28a4-11e9-9102-a177f089ce6f.PNG)
![camera_calib_codesnapshot_4](https://user-images.githubusercontent.com/25223180/52207634-054a1700-28a4-11e9-9739-9cbd22d7df15.PNG)

The code for this step is contained in the above displayed Code Cells of the Jupyter Notebook (Advanced_Lane_Detection.ipynb) located 
in CarND-Advanced-Lane-Lines/ folder
## Explanation of the logic for this step 
Camera Calibration is required as the camera distorts the shape and size of 3-D objects as they are captured by camera lens in a 2-D 
Frame . In order to implement this the 3-D realtime Coordinates of a known object need to be mapped to the 2-D Coordinates of the Camera
Frame. 2 separate lists 1) "imgpoints" for containing the  2-D coordinates of the frame and 2)"Objpoints" for containing the 3-D 
coordinates of the corrseponding realtime objects are created for this. The Object Points would remain constant for every calibration
image with z-value being 0 . The variable "objp" is a replicated array of coordiantes appended every time to the "Objpoints" list.
For finding the 2-D Coordiantes of the 2-D image of the 3-D object first of all the image is converted to grayscale and is passed as
argument to the function cv2.findChessboardCorners() along with the values nx:number of inside corners along x-axis and ny:number of inside corners along y-axis . If this returns some corners detected in the image then these corners are appended to the "imgpoints" list
Now that the 2-D corners corresponding to the 3-D corner values are available then these corners are drawn on the image using the 
function cv2.drawChessboardCorners(). Hence now that both the imgpoints and objpoints lists are available corresponding to the 2-D
corners detected and the static 3-D corner values these are passed as argumets to the function cv2.calibrateCamera() alongwith the shape
of the grayscale image. This fucntion returns the following values : Distortion Coefficient / Camera Matrix (required to transform 3-D
object points to 2-D Image points ) / Rotation and Translation Vectors (telling about the position of the camera in the world)
Now in the code the function cv2.undistort() takes in the distorted image / camera matrix and distortion coefficient as arguments and
returns the undistorted image also termed as destination image 

Below are the results achieved by applying the above algorithm to undistort images 
![undistorted_checkboard12](https://user-images.githubusercontent.com/25223180/52215660-87ddd100-28ba-11e9-8304-eb9547267867.PNG)
![undistorted_testimg4](https://user-images.githubusercontent.com/25223180/52209300-835cec80-28a9-11e9-9842-48c0fe64ddc3.PNG)

### 2. Threshold Binary Image Generation
### Code Snapshots

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  

The code for this step is contained in the above displayed Code Cells of the Jupyter Notebook (Advanced_Lane_Detection.ipynb) located 
in CarND-Advanced-Lane-Lines/ folder
## Explanation of the logic for this step 

Below are the results achieved by applying the above algorithm to generate Threshold Binary Image on Undistorted Images obtained from previous step.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
