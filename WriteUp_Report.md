# **Behavioral Cloning** 
# Problem Statement Description
For this project the task is to accurately predict the steering angle to be taken by the simulated car based on the inputs of LIVE
Camera Feed of Center , Left and Right Cameras mounted on top of the Car and hence clone / imitate the driving behaviour of a real person behind the steering wheel.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

# Details of running the Code Implementation :
  Using the self-drive car Unity simulator and drive.py (Version 1) / drive_V2.py (Version 2) file, the car can be driven autonomously
  around the track by the following steps:
  1) First of all clone this repository (https://github.com/PranayKr/Self-Driving-Car-Projects.git) on local system.
  2) Open Anaconda prompt shell window and navigate inside the Behavior-Cloning/ cloned repository folder.
  3) Start up the self-driving simulator, choose a scene and press the Autonomous Mode button.
  3) Run the pretrained model by executing the following command : 
     
     a) Version 1 
     ```sh
     python drive.py model.h5
     ```
     b) Version 2
     ```sh
     python drive_V2.py model_updated.h5
     ```

### Model Architecture and Training Strategy
For this project a basic Convolutional Neural Network Architecture has been used developed using Keras Library which is trained on LIVE
Feeds of Cameras mounted on top of the car-agent in the Unity Simulation environment. The strategy used is that of using Supervised
Image Classification / Regression algorithm where the ConvNet Model trained on images of the inputs provided by the
central,rigt and left cameras is able to accurately predict what steering angle the simulated car should take so that it always remains
on the lane and nevers veers off the track / road to surrounding country side simulated in the environment provided.

The design of the network is inspired from the state-of-the-Art NVIDIA model, which has been used by NVIDIA for the end-to-end self driving test by mapping the raw pixels from a front-facing camera to the steering commands for a self-driving car.
<table>
  <tr><td align="center">NVIDIA CNN Model</td></tr>
  <tr><td><image src="https://user-images.githubusercontent.com/25223180/53684690-74494d00-3d36-11e9-9f14-8a0cfad23e39.png"</td></tr>
</table>
The Convolutional Neural Net Model created has multiple Convolutional Layers followed by few feedforward Neural-Net Layers.
The Convoluyional Layers act as feature-learners learning the unique features of the lanes images provided as camera feed whereas
the fully connected layers (feedforward dense layers) take up the task of executing logistic regression algorithm on the basis of the 
output of Convolutional Layers to predict steering angle label's value provided a certain Image as input.

Hence During Training the Training data has been divided into features (Input Feed Images procured from the Central,Right and Left
Cameras) and lables (Speed / Throttle / Steering and Brake values) out of which basically the model needs to predict the value of the
lable "Steering Angle"

Efforts were made to develop such a ConvNet Model and improvise on its efficiency and accuracy by trying the below mentioned details
of 2 different versions of CNN Architecture

<table>
  <tr><td colspan=2 align="center">CNN Model Architecture Summary (using Keras Library)</td></tr>
  <tr><td align="center">Version 1</td><td align="center">Version 2</td></tr>
  <tr>
  <td><image src="https://user-images.githubusercontent.com/25223180/53684231-ed45a600-3d30-11e9-8eed-eec0c82d87f8.PNG"></td>
  <td><image src="https://user-images.githubusercontent.com/25223180/53684235-f3d41d80-3d30-11e9-961d-921f750e7ef8.PNG"></td>
  </tr>
</table>

<table>
  <tr><td colspan=2 align="center">CNN Model Architecture Desciption and Comparison</td></tr>
  <tr><td align="center">Version 1</td><td align="center">Version 2</td></tr>
  <tr>
    <td><table><tr><td>Lambda Layer for input images normalization (dimension of input images (160, 320, 3))</td></tr>
      <tr><td>Cropping2D layer to resize images to remove the top pixels having sky and trees and landscape and bottom part having the
      front bonet of the car in view</td></tr><tr><td>2 Convolution Layers having SAME padding,ELU Activation function 
      and kernel size of 5x5</td></tr><tr><td>1 Convolution Layer having VALID padding,ELU Activation function 
      and kernel size of 5x5</td></tr><tr><td>2 Convolution Layers having VALID padding,ELU Activation function 
      and kernel size of 3x3</td></tr><tr><td>One Flatten Layer</td></tr><tr><td>One Dropout Layer with probability of 0.5 to avoid
      overfitting</td></tr><tr><td>3 Dense Layers with ELU Activation function</td></tr><tr><td>1 Dense Layer with single output Neuron 
      to predict the Steering Angle Value</td></tr><tr><td>ADAM Optimizer used with Learning Rate of 1.0e-4 (0.0001)</td></tr><tr>
      <td>Mean Square Error Loss Function used</td></tr></table></td>
  <td><table><tr><td>Lambda Layer for input images normalization (dimension of input images (66, 200, 3))</td></tr><tr><td>3 Convolution
      Layers having ELU Activation function and kernel size of 5x5</td></tr><tr><td>2 Convolution
      Layers having ELU Activation function and kernel size of 3x3</td></tr><tr><td>Dropout Layer with probability of 0.5 to avoid
      overfitting</td></tr><tr><td>One Flatten Layer</td></tr><tr><td>3 Dense Layers with ELU Activation function</td></tr><tr><td>1 
      Dense Layer with single output Neuron to predict the Steering Angle Value</td></tr><tr><td>ADAM Optimizer used with Learning Rate 
      of 1.0e-4 (0.0001)</td></tr><tr><td>Mean Square Error Loss Function used</td></tr></table></td>
  </tr>
</table>
<table>
  <tr><td colspan=3 align="center">HYPERPARAMETERS</td></tr>
  <tr><td></td><td align="center">Version 1</td><td align="center">Version 2</td></tr>
  <tr><td>INPUT_SHAPE</td><td>(160, 320, 3)</td><td>(66, 200, 3)</td></tr>
  <tr><td>BATCH_SIZE</td><td>32</td><td>40</td></tr>
  <tr><td>NUMBER OF EPOCHS</td><td>15</td><td>10</td></tr>
  <tr><td>LEARNING RATE</td><td>1.0e-4</td><td>1.0e-4</td></tr>
  <tr><td>STEERING ANGLE CORRECTION (ADJUSTMENT)</td><td><table><tr><td>+ 0.25</td><td>LEFT IMAGE (+)</td></tr><tr><td>- 0.25</td>
  <td>RIGHT IMAGE (-)</td></tr></table></td><td><table><tr><td>+ 0.2</td><td>LEFT IMAGE (+)</td></tr><tr><td>- 0.2</td><td>RIGHT
  IMAGE (-)</td></tr></table></td></tr>
  <tr><td>DROPOUT PROBABILITY</td><td>0.5</td><td>0.5</td></tr>
  <tr><td>LOSS FUNCTION</td><td>MEAN SQUARE ERROR</td><td>MEAN SQUARE ERROR</td></tr>
  <tr><td>OPTIMIZER</td><td>ADAM OPTIMIZER</td><td>ADAM OPTIMIZER</td></tr>
</table>

### DATA PREPROCESSING AND DATA AUGMENTATION 
#### Creation of the Training Set & Training Process
The Sample Training Data was used for both the versions of the Deep ConvNet Models
The Data was split into Training and Validation Set to measure the performance of the CNN model at every epoch.
Adam optimizer was used for optimization with learning rate of 1.0e-4 and MSE loss function (MEAN SQUARE ERROR)
<table>
   <tr><td colspan=2 align="center">DATA PREPROCESSING</td></tr>
   <tr><td align="center">Version 1</td><td align="center">Version 2</td></tr>
   <tr><td>The images are cropped using Cropping2D layer in ConvNet Model to remove the top and bottom pixels having the sky and the car 
   front parts </td><td>Images are cropped so that the model won’t be trained with the sky and the car front parts</td></tr>
   <tr><td>Images are not resized but on cropping the input dimensions reduce</td><td>Images are resized to 66x200 (3 YUV channels) 
   as per NVIDIA model</td></tr><tr><td>Images are normalized (image data divided by 255.0 and subtracted 0.5) to avoid saturation and 
   make gradients work better</td><td>Images are normalized (image data divided by 127.5 and subtracted 1.0) to avoid saturation and 
   make gradients work better</td></tr>    
</table>
<table>
  <tr><td colspan=2 align="center">DATA AUGMENTATION</td></tr>
  <tr><td align="center">Version 1</td><td align="center">Version 2</td></tr>
  <tr><td><table><tr><td><table><tr><td colspan=3 align=c"enter">Random Selection of Left/Right or Center Images with STEERING ANGLE 
  CORRECTION value as (+/-) 0.25</td></tr><tr><td>STEERING ANGLE CORRECTION</td><td>LEFT IMAGE</td><td>(+) 0.25</td></tr><tr>
  <td>STEERING ANGLE CORRECTION</td><td>RIGHT IMAGE</td><td>(-) 0.25</td></tr></table></td><td><tr><td>ROW2</td></tr></table></td></tr>
  </table></td><td><table><tr><td colspan=3 align=c"enter">Random Selection of Left/Right or Center Images with STEERING ANGLE 
  CORRECTION value as (+/-) 0.2</td></tr><tr><td>STEERING ANGLE CORRECTION</td><td>LEFT IMAGE</td><td>(+) 0.2</td></tr><tr><td>STEERING 
  ANGLE CORRECTION</td><td>RIGHT IMAGE</td><td>(-) 0.2</td></tr></table></td></tr></td></tr></table>
  
  
   
 
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.







