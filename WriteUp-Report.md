# **Finding Lane Lines on the Road** 

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. PIPELINE DESCRIPTION
The Software Pipeline built for Lane Detection consists of the following steps:
1) Get the GreyScale Version of the Original Image
2) Apply Gaussian Smoothing with a ser kernel size value to remove noise from the GreyScaled Image
3) Apply Canny Edge Detection with upper and lower threshold values to select edges with strong gradients
4) Get the ROI(region of interest) corresponding to the Road Lane Lines in the processed image by masking the image using 
   cv2.fillPoly() function
5) Run Hough transform Algorithm using the required parameters on masked edge-detected image
6) Draw line segments
7) Extrapolate and Average the Hough line segments corresponding to Left Lane and Right Lane by taking into consideration their slopes
   and intercepts
8) Draw lines corresponding to the Left Lane and Right Lane by getting the coordinates from their respective slopes and intercepts.
8) Combine Hough lines image with original image to verify the accuracy of the annotated lane lines.

### 2. RESULTS SHOWCASE

Graphs of the Image (solidWhiteCurve.jpg) showcasing the Pipeline Results at various stages

![image1_greyscale](https://user-images.githubusercontent.com/25223180/51797231-a80bf100-2225-11e9-8263-6d88e6382fe2.PNG)

![image1_houghtrnsfrm](https://user-images.githubusercontent.com/25223180/51797240-d12c8180-2225-11e9-8453-a35f89284359.PNG)

![image1_finalresult](https://user-images.githubusercontent.com/25223180/51797253-fc16d580-2225-11e9-8157-ae10b035d974.PNG)



### 3. POTENTIAL SHORTCOMINGS
The Lane Detection Algorithm implemented has the below mentioned shortcomings :
1) The Algorithm would work properly only when the Camera is placed at a fixed position on top
   of the car's hood . If the position of the camera is changed such that it is not directly facing the road ahead but at a certain angle 
   or tilt then the same results won't be achieved.
2) If the resolution of the camera is changed accordingly the logic of the algorithm would have to be modified to get the results.
3) The Lighting conditions in the LIVE Feed of the camera would affect the results
4) If the weather conditons change from a clear sunny day to a stormy / snowy / rainy / cloudy day the results obtained would get
   affected.
5) For LIVE Feed having Lane Lines having high curvature instead of being straight i.e. at turning points on the road the algorithm
   would not be that accurate in lane detection with its efficiency inversely proportional to the sharpness of turns on road.


### 4. POSSIBLE IMPROVEMENTS

1) A generalized version of the current algorithm having a pipeline which can be dynamically optimized to do away with the dependencies 
   required for getting the proper results would have to be developed which would be invariant to :- 
   a) Camera's Position and Angle / Orientation  
   b) Camera's Resolution 
   c) Lighting Conditions 
   d) Weather Conditions 
   
2) Further finetuning of the Hough Transform Algorithm to detect lines edges with more variety of images having lanes in diffrent
   regions and lighting conditions / weather conditons / camera position and quality (resolutions / angles / orientations)
   
3) Building a Lane Detection Algorithm using Computer Vision Techniques (ConvNet Models) trained on a dataset of lane markings on road
   instead of using the general Image Processing Algorithms used in this implementation



