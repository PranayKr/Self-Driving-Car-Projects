# **Finding Lane Lines on the Road** 

[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. PIPELINE DESCRIPTION

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. POTENTIAL SHORTCOMINGS
The Lane Detection Algorithm implemented has the below mentioned shortcomings :
1) The Algorithm would work properly only when the Camera is placed at a fixed position on top
   of the car's hood . If the position of the camera is changed such that it is not directly facing the road ahead but at a certain angle 
   or tilt then the same results won't be achieved.
2) If the resolution of the camera is changed accordingly the logic of the algorithm would have to be modified to get the results.
3) The Lighting conditions in the LIVE Feed of the camera would affect the results
4) If the weather conditons change from a clear sunny day to a stormy / snowy / rainy / cloudy day the results obtained would get affected


### 3. POSSIBLE IMPROVEMENTS

1) A generalized version of the current algorithm having a pipeline which can be dynamically optimized to do away with the dependencies 
   required for getting the proper results would have to be developed which would be invariant to the 
   1) Camera's Position and Angle / Orientation  2) Camera's Resolution 3) Lighting Conditions 4) Weather Conditions 
   
2) Building a Lane Detection Algorithm using Computer Vision Techniques (ConvNet Models) trained on a dataset of lane markings on road
   instead of using the general Image Processing Algorithms used in this implementation



