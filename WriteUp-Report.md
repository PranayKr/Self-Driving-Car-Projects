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

The code for this step is contained in the above displayed Code Cells (Cell numbers 2,3,4,5,6,7,8,9,10,11) of the Jupyter Notebook
(Advanced_Lane_Detection.ipynb) located in CarND-Advanced-Lane-Lines/ folder and lines 38-154 of the python file
Advanced_Lane_Detection.py) located in CarND-Advanced-Lane-Lines/ folder 

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
![code_snippet_1](https://user-images.githubusercontent.com/25223180/52234547-f08f7280-28e7-11e9-89eb-04366c897911.PNG)
![code_snippet_2](https://user-images.githubusercontent.com/25223180/52234554-f7b68080-28e7-11e9-9970-3bd1f638379c.PNG)
![code_snippet_3](https://user-images.githubusercontent.com/25223180/52234558-fbe29e00-28e7-11e9-8267-61263018f550.PNG)
![code_snippet_4](https://user-images.githubusercontent.com/25223180/52234568-ff762500-28e7-11e9-9ae1-3be24ee8c0d8.PNG)
![code_snippet_5](https://user-images.githubusercontent.com/25223180/52234576-02711580-28e8-11e9-8bfd-ec7a534f75c8.PNG)

The code for this step is contained in the above displayed Code Cells (Cell numbers 12,13,14) of the Jupyter Notebook
(Advanced_Lane_Detection.ipynb) located in CarND-Advanced-Lane-Lines/ folder and lines 158-272 of the python file
Advanced_Lane_Detection.py) located in CarND-Advanced-Lane-Lines/ folder 
## Explanation of the logic for this step
Gradient Threshold Calculations are required to isolate only the pixels corresponding to Lane Edges in a given undistorted image instead
of directly using Canny Edge Detection method which outputs edges of surrounding background and other objects such as vehicles/ trees/
buildings as well. Canny Edge Detection internally uses Sobel-x and Sobel-y operators to calculate gradients along the x and y-axes.
For implementation of this combined gradient threshold is calculated combining the results of sobel-x and sobel-y operators , magnitude
threshold and direction threshold.

For getting gradient threshold along x and y directions the folliwng steps are used :
1) The undistorted image is converted to grayscale
2) cv2.Sobel() fucntion is applied
3) The absolute value of the derivative or gradient calculated using cv2.Sobel() fucntion is calculated using np.absolute()
4) The absolute value is scale to 8-bit (0 - 255) then convert to type = np.uint8
5) Lower and upper thresholds of values 20 and 100 are applied on the scaled absolute gradient
6) This masked output is returned as binary output 
Magnitude Gradient threshold is calculated in a similar way the only difference being that first of all the magnitude gradient is 
calculated by taking the square root of the sum of the squares of gardients/ derivatives along x and y-directions (outputs of Sobel-x
and Sobel-y operators)
Direction Gradient Threshold is calculated by taking the inverse tangent of quotient of absolute gradient over y-direction over the 
absolute gradient over x-direction and applying a mask using the lower threshold value of 0.7 and upper threshold value of 1.4 to get
the binary output.

Now once the combined gradient threshold is available Color Gradient Threshold output is also calculated by the following steps:
The undistored image is converted used the function cv2.cvtColor() first to RGB color space and then to HLS Color Space. The 
S- Channel of the converted image (Saturation cahnnel)  is isolated to apply a mask using lower color threshold value of 150 and upper 
color threshold value of 255

Now both the combined gradient threshold output and color threshold outputs are combined to generate the final threshold binary image
with the Lane Edges pixels isloted in the entire image to maximum possibel extent.

Below are the results achieved by applying the above algorithm to generate Threshold Binary Image on Undistorted Image obtained from previous step.
![binary_threshimg_2](https://user-images.githubusercontent.com/25223180/52237577-ee311680-28ef-11e9-9040-f52fe21ef424.PNG)

### 3. Perspective Transform Step
### Code Snapshots
![code_snippet_1](https://user-images.githubusercontent.com/25223180/52238376-d3f83800-28f1-11e9-8b5b-0143b015e43a.PNG)
![code_snippet_2](https://user-images.githubusercontent.com/25223180/52238382-d8245580-28f1-11e9-8de0-bc57572100e1.PNG)
![code_snippet_3](https://user-images.githubusercontent.com/25223180/52238392-db1f4600-28f1-11e9-9605-379066af9466.PNG)
The code for this step is contained in the above displayed Code Cells (Cell numbers 15,16,17) of the Jupyter Notebook
(Advanced_Lane_Detection.ipynb) located in CarND-Advanced-Lane-Lines/ folder and lines 277-323 of the python file
Advanced_Lane_Detection.py) located in CarND-Advanced-Lane-Lines/ folder 
## Explanation of the logic for this step
A perspective transform maps the points in a given image to different, desired, image points with a new perspective. The perspective
transform relevant for this implementation is a bird’s-eye view transform that let’s us view a lane from above which would be used for
calculation of the lane curvature. Perpective Trsnform helps us to see the same scene with different viewpoints and angles.
The process of applying  a Perspective Transform to an image is similar to the method of correcting distortion in an image but the
difference is that while in distortion correction 3-D object points are mapped 2-D image points ; in case of Perspective transform
2-D Image points are mapped to different set of 2-D Image points with a new perspective

Following are the steps to apply perspective tranform on the generated Binary Threshold Image from previous step:
1) 4 source image points were taken on the threshold binary image which define a rectangular plane 
2) 4 destination image points were selected which define a rectangular plane to map the source image points on the warped transformed
   image. 
   
   This resulted in the following source and destination points:  
   a) | Source      | Destination   | 
   b) |:-----------:|:-------------:| 
   c) | 568,470     | 200,0         | 
   d) | 717,470     | 200,680       |
   e) | 260,680     | 1000,0        |
   f) | 1043,680    | 1000,680      |
   
3) Now both the source image points and destination image points are passed to a function cv2.getPerspectiveTransform() to get
   Perspective Transform Matrix
4) The warped image i.e. the prespective transformed iamge is now generated by calling the function cv2.warpPerspective() to which
   the binary threshold image , Perspective Transform Matrix from previous step , the size of the warped image to be generated and 
   a flag cv2.INTER_NEAREST to interpolate the missing points / coordinates while generating the warped image are passed as arguments

Below are the results achieved by applying the above algorithm to generate Perspective Transformed Image on generated Binary Threshold 
Images obtained from previous step.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  


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
