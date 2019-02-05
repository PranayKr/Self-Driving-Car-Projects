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

### NOTE : All the Code snapshots are from the implementation done in the jupyter notebook file Advanced_Lane_Detection.ipynb file which could not be uploaded in github owing to large size (> 25 mb) as it contained the results for all the test images at various stages of implementation. Instead the jupyter notebook file Advanced_Lane_Detection-Updated.ipynb was uploaed in github which contains the rsults generated on a single test image during various stages of implementation of the below described pipeline. Owing to lack of time I am unable to replace the provided code snapshots from the file Advanced_Lane_Detection.ipynb with the code snapshots from the actually uploaded implementation file Advanced_Lane_Detection-Updated.ipynb. Logically the code in both the files are the same except for one extra condition logic to visualize results only for a single test image in the file Advanced_Lane_Detection-Updated.ipynb instead of iterating over all the test images to display reulst / plot graphs for all provided test images as was doen in the case of the code in the larger not-uploaded file Advanced_Lane_Detection.ipynb. 

### NOTE : Please consider the above discripancy. Eventually I will upload the proper code snapshots reflecting the implemenation in actually uploaded jupyter notebook file. Besides the source code in the python file (.py) implementation of both versions (both.ipynb files) have been uploaded in this repository for reference.          

### Camera Calibration and Distortion Correction
### Code Snapshots
![camera_calib_codesnapshot_](https://user-images.githubusercontent.com/25223180/52207620-fb281880-28a3-11e9-87b1-9dd46d3a887f.PNG)
![camera_calib_codesnapshot_2](https://user-images.githubusercontent.com/25223180/52207625-ff543600-28a3-11e9-8e37-85b084195797.PNG)
![camera_calib_codesnapshot_3](https://user-images.githubusercontent.com/25223180/52207629-024f2680-28a4-11e9-9102-a177f089ce6f.PNG)
![camera_calib_codesnapshot_4](https://user-images.githubusercontent.com/25223180/52207634-054a1700-28a4-11e9-9739-9cbd22d7df15.PNG)

The code for this step is contained in the above displayed Code Cells (Cell numbers 2,3,4,5,6,7,8,9,10,11) of the Jupyter Notebook
(Advanced_Lane_Detection-Updated.ipynb) located in CarND-Advanced-Lane-Lines/ folder and lines 38-154 of the python file
Advanced_Lane_Detection.py) located in CarND-Advanced-Lane-Lines/ folder 

NOTE: Advanced_Lane_Detection.ipynb was too large (>25 mb) to be uploaded on github

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
(Advanced_Lane_Detection-Updated.ipynb) located in CarND-Advanced-Lane-Lines/ folder and lines 158-272 of the python file
Advanced_Lane_Detection.py) located in CarND-Advanced-Lane-Lines/ folder

NOTE: Advanced_Lane_Detection.ipynb was too large (>25 mb) to be uploaded on github

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
(Advanced_Lane_Detection-Updated.ipynb) located in CarND-Advanced-Lane-Lines/ folder and lines 277-323 of the python file
Advanced_Lane_Detection.py) located in CarND-Advanced-Lane-Lines/ folder

NOTE: Advanced_Lane_Detection.ipynb was too large (>25 mb) to be uploaded on github

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
   image. This resulted in the following source and destination points:  
   ```python
   | Source      | Destination   | 
   |:-----------:|:-------------:| 
   | 568,470     | 200,0         | 
   | 717,470     | 200,680       |
   | 260,680     | 1000,0        |
   | 1043,680    | 1000,680      |
   ```   
3) Now both the source image points and destination image points are passed to a function cv2.getPerspectiveTransform() to get
   Perspective Transform Matrix
4) The warped image i.e. the prespective transformed iamge is now generated by calling the function cv2.warpPerspective() to which
   the binary threshold image , Perspective Transform Matrix from previous step , the size of the warped image to be generated and 
   a flag cv2.INTER_NEAREST to interpolate the missing points / coordinates while generating the warped image are passed as arguments

Below are the results achieved by applying the above algorithm to generate Perspective Transformed Images on generated Binary Threshold 
Images obtained from previous step
![perspectivetrnsfrm_img2](https://user-images.githubusercontent.com/25223180/52241236-1ec97e00-28f9-11e9-9f2a-704386d85238.PNG)
![perspectivetrnsfrm_img3](https://user-images.githubusercontent.com/25223180/52241242-225d0500-28f9-11e9-86f2-1233a4a11306.PNG)
As is evident from the above results that the lane lines appear parallel in the warped images thus validating that the perspective
transform logic is working as expected.
### 4. Lane-Line pixels identification in the Warped threshold binary image using Histogram Peak Calculation and Sliding Windows Approach followed by fitting the pixels with a 2nd-order polynomial
### Code Snapshots
![code_snippet_1](https://user-images.githubusercontent.com/25223180/52242445-b8465f00-28fc-11e9-8dd7-54e11a36c837.PNG)
![code_snippet_2](https://user-images.githubusercontent.com/25223180/52242451-bbd9e600-28fc-11e9-8080-a84e36adb902.PNG)
![code_snippet_3](https://user-images.githubusercontent.com/25223180/52242455-c0060380-28fc-11e9-832f-c077eff47ea3.PNG)
![code_snippet_4](https://user-images.githubusercontent.com/25223180/52242459-c3998a80-28fc-11e9-9f8d-2d4c00c67a2a.PNG)
![code_snippet_5](https://user-images.githubusercontent.com/25223180/52242466-c72d1180-28fc-11e9-9083-5c8997a80ba9.PNG)
![code_snippet_6](https://user-images.githubusercontent.com/25223180/52242472-cb592f00-28fc-11e9-8f9b-69304a043d00.PNG)
![code_snippet_7](https://user-images.githubusercontent.com/25223180/52242480-ce541f80-28fc-11e9-9734-51deac774920.PNG)

The code for this step is contained in the above displayed Code Cells (Cell numbers 18,19,20) of the Jupyter Notebook
(Advanced_Lane_Detection-Updated.ipynb) located in CarND-Advanced-Lane-Lines/ folder and lines 327-499 of the python file
Advanced_Lane_Detection.py) located in CarND-Advanced-Lane-Lines/ folder

NOTE: Advanced_Lane_Detection.ipynb was too large (>25 mb) to be uploaded on github

## Explanation of the logic for this step
In order to determine which pixels in the warped threshold binary image belong to the left lane and which pixels belong to the right
lane a histogram along all the columns of  bottom half of the image is calculated. With this calculation all the pixel values along each
column in the image are summed up. Since the pixel values are either 0 or 1 the two most prominent peaks in this histogram act be good
indicators of the x-position of the bottom of the lane lines. Taking that as a starting point for searching for the pixels belonging to
the right and left lanes a sliding window i.e. a fixed-size rectangle , placed around the line centers is used to find and follow the
lines up to the top of the frame. Once the pixels belonging to each lane line are identified the built-in function np.polyfit() method fits a second order polynomial (f(y) = Ay² + By + C) to each set of pixels.

The Detect_lines_beyond() function performs the same task but optimizes the searching of lane-line pixels by using a previous
fit (from a previous image frame of the LIVE video feed) and only searching for lane pixels within a certain range of that fit.

Below are the results achieved by applying the above algorithm to show the Lane Lines Identified on the Warped Binary Threshold 
Images obtained from previous step. The green shaded area is the range from the previous fitted polynomial over the lane pixels
detected, and the yellow lines are from the current image.

![lanes_detected_1](https://user-images.githubusercontent.com/25223180/52251012-bc847380-2920-11e9-9b67-78499c373af4.PNG)
![lanes_detected_2](https://user-images.githubusercontent.com/25223180/52251015-bf7f6400-2920-11e9-83ad-6df4f43b344a.PNG)
![lanes_detected_3](https://user-images.githubusercontent.com/25223180/52251017-c27a5480-2920-11e9-93c5-2852ef91daab.PNG)
![lanes_detected_4](https://user-images.githubusercontent.com/25223180/52251019-c4dcae80-2920-11e9-8d9a-f5c69917eadd.PNG)
![lanes_detected_5](https://user-images.githubusercontent.com/25223180/52251024-c908cc00-2920-11e9-85a3-676630817577.PNG)
![lanes_detected_6](https://user-images.githubusercontent.com/25223180/52251027-cc9c5300-2920-11e9-892f-509eb5e52824.PNG)
### 5. Calculation of the Radius of Curvature of the Left and Right Lanes and the position of the vehicle with respect to center.
### Code Snapshots
![code_snippet_1](https://user-images.githubusercontent.com/25223180/52251677-513ca080-2924-11e9-8a01-1e3071dd4e70.PNG)
![code_snippet_2](https://user-images.githubusercontent.com/25223180/52251682-54d02780-2924-11e9-9794-860adb1b1392.PNG)

The code for this step is contained in the above displayed Code Cells (Cell number 21) of the Jupyter Notebook
(Advanced_Lane_Detection-Updated.ipynb) located in CarND-Advanced-Lane-Lines/ folder and lines 504-536 of the python file
Advanced_Lane_Detection.py) located in CarND-Advanced-Lane-Lines/ folder 

NOTE: Advanced_Lane_Detection.ipynb was too large (>25 mb) to be uploaded on github

## Explanation of the logic for this step
The calculation of radius of curvature is done as explained in this tutorial 
(ref link: https://www.intmath.com/applications-differentiation/8-radius-curvature.php)

A brief explanation of this concept can be understood by referring to the below provided image

![radius of curavture equations](https://user-images.githubusercontent.com/25223180/52251862-51896b80-2925-11e9-9223-f8980efbc671.PNG)

Now taking into consideration the above provided expalnation the radius of curvature is calculated by the following lines of code
in the implementation

![radiiofcurve_eqns](https://user-images.githubusercontent.com/25223180/52252092-6fa39b80-2926-11e9-8ac7-067d5854b283.PNG)

In this example, left_fit_cr[0] and right_fit_cr[0] are the first coefficients (the y-squared coefficient) of the second order
polynomial fit, whereas left_fit_cr[1] and right_fit_cr[1] are the second (y) coefficient. y_eval is the y position within the image
upon which the curvature calculation is based (the bottom-most y - the position of the car in the image - was chosen). y_m_per_pix is
the factor used for converting from pixels to meters. This conversion was also used to generate a new fit with coefficients in terms of
meters.

The position of the vehicle with respect to the center of the lane is calculated with the following lines of code:

![vehicle_center_eqn](https://user-images.githubusercontent.com/25223180/52252097-74684f80-2926-11e9-9263-33df9acb2edc.PNG)
![lane_center_eqn](https://user-images.githubusercontent.com/25223180/52252103-7f22e480-2926-11e9-9854-10f08e9de9eb.PNG)

left_lane_bottom and right_lane_bottom are the x-intercepts of the left and right fits, respectively. The deviation of the
vehicle from the center is calculated by taking the difference of lane center (i.e. midpoint of the sum of intercepts of left
fit and right fit) and center of image i..e. 640 in this case multimplied by the scaling factor "xm_per_pix" to get the real-time
position from the pixel position.

### 6. Results plotted back down onto the road such that the lane area is identified clearly.
### Code Snapshots
![code_snippet_1](https://user-images.githubusercontent.com/25223180/52252779-5f8dbb00-292a-11e9-9f6d-d3001c75485e.PNG)
![code_snippet_2](https://user-images.githubusercontent.com/25223180/52252781-6288ab80-292a-11e9-947e-13fc87fbd620.PNG)

The code for this step is contained in the above displayed Code Cells (Cell number 22,23,24,25,26,27,28)of the Jupyter Notebook
(Advanced_Lane_Detection-Updated.ipynb) located in CarND-Advanced-Lane-Lines/ folder and lines 539-700 of the python file
Advanced_Lane_Detection.py) located in CarND-Advanced-Lane-Lines/ folder

NOTE: Advanced_Lane_Detection.ipynb was too large (>25 mb) to be uploaded on github

Below are the results achieved by plotting back down onto the road such that the lane area is identified clearly.
![finalimageresult1](https://user-images.githubusercontent.com/25223180/52252841-9e237580-292a-11e9-95e9-5ec689585d36.PNG)
![finalimageresult2](https://user-images.githubusercontent.com/25223180/52252848-a5e31a00-292a-11e9-910e-b973fbaf227a.PNG)
![finalimageresult3](https://user-images.githubusercontent.com/25223180/52252852-aa0f3780-292a-11e9-8f2d-3085cc4a6fff.PNG)
![finalimageresult4](https://user-images.githubusercontent.com/25223180/52252858-ae3b5500-292a-11e9-9535-d450eea6ea55.PNG)
![finalimageresult5](https://user-images.githubusercontent.com/25223180/52252862-b1cedc00-292a-11e9-9c14-d2bae86121e1.PNG)
![finalimageresult6](https://user-images.githubusercontent.com/25223180/52252865-b5faf980-292a-11e9-8ff2-25be7c17680d.PNG)

### Pipeline (video)
Animated GIF image showing the results obtained on a section of the entire video

![advanced-lane-detection-result](https://user-images.githubusercontent.com/25223180/52253015-6ec13880-292b-11e9-9e5b-231bae1a8f5a.gif)

Here's a [link to the entire video result of Advanced Lane Detection Algorithm Implementation](https://youtu.be/WUKgYlx1qU0)

### Discussion
### Problems / Issues faced in implementation of this Project
### PIPELINE FAILURE 
1) The Advanced Lane Detection Algorithm Implementation fails to detect lanes properly in the challenge video (challenge_video.mp4)
   and harder challenge video (harder_challenge_video.mp4)
   
   One of the reasons for failure was the low lighting conditions present in the harder challenge video (harder_challenge_video.mp4)
   and sharper turns appearing consistently over the meandering roads in the video
   
   The implementation would also probably not work as expected if provided a video of vehicle driving down in snowy/rainy/stormy
   weather conditions or if the LIVE Feed is from a vehicle driving down a road during night time.
      
2) The execution time of implementation of the various image processing algorithm was very high due to slower processing power of 
   my CPU-Laptop. The execution time should reduce significantly if the code is executed on GPU-enabled systems with considerably 
   higher RAM.
   
### POSSIBLE IMPROVEMENTS 
1) Better Hyperparameter tuning to get more appropriate gradient thresholds for lane pixels detection
2) More finetuning / tweaking of parameters to get better Radius of Curvature Estimates 
3) Implementation of advanced lane detection using self-learning Computer Vision Techniques such as training Deep Learning Neural Net
   Models (Convolutional Neural Nets) instead of using the conventional Image Processing Algorithms 

### REFERENCES
EXPLANATION OF THE CONCEPT OF RADIUS OF CURVATURE AND ITS IMPLEMENTATION 
(https://www.intmath.com/applications-differentiation/8-radius-curvature.php)

