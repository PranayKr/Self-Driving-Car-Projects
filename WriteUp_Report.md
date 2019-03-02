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

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

