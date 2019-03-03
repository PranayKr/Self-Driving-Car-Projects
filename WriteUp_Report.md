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
The Convolutional Layers act as feature-learners learning the unique features of the lanes images provided as camera feed whereas
the fully connected layers (feedforward dense layers) take up the task of executing logistic regression algorithm on the basis of the 
output of Convolutional Layers to predict steering angle label's value provided a certain Image as input.

Hence During Training the Training data has been divided into features-set (Input Feed Images procured from the Central,Right and Left
Cameras) and lables-set (Steering Angle Values) Basically the model trained on the Left/Right/Center Camera Input feeds (Images)
needs to predict the value of the lable "Steering Angle".
The summary of the values of all the 4 parameters provided along the Images Feed (corresponding to 3 cameras):
1) The Steering Angle measurements range between -1 to 1
2) The Throttle measurments range between 0 to 1
3) The Brake measurments all happen to be 0
4) The Speed measuremnts range between 0 to 30 

Efforts were made to develop such a ConvNet Model and improvise on its efficiency and accuracy by trying the below mentioned details
of 2 different versions of CNN-Model Architecture

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
The Sample Training Data-Set (corresponding to the Lake Track Simulation Environment) enhanced / modified by relatively different 
Data-Augmentation techniques was used for training both the versions of the Deep ConvNet Models.
(Version 1 (model.h5) and Version 2 (model_updated.h5)).
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
  <tr><td><table><tr><td><table><tr><td colspan=3 align="center">Random Selection of Left/Right or Center Images with STEERING ANGLE 
  CORRECTION value as (+/-) 0.25 for Left/Right Images and 0 for Center Images</td></tr><tr><td>STEERING ANGLE CORRECTION</td><td>LEFT 
  IMAGE</td><td>(+) 0.25</td></tr><tr>
  <td colspan=3><Image src="https://user-images.githubusercontent.com/25223180/53690842-3da41e80-3d98-11e9-9de7-47601ebeeab1.PNG"></td>
  </tr><tr><td>STEERING ANGLE CORRECTION</td><td>RIGHT IMAGE</td><td>(-) 0.25</td></tr><tr>
  <td colspan=3><Image src="https://user-images.githubusercontent.com/25223180/53690866-bacf9380-3d98-11e9-9769-178f7bc6779b.PNG"></td>
  </tr><tr><td>STEERING ANGLE CORRECTION</td><td>CENTER IMAGE</td><td>0 (NOT REQUIRED)</td></tr><tr><td colspan="3"> 
  <Image src="https://user-images.githubusercontent.com/25223180/53691245-8f9c7280-3d9f-11e9-9bec-b7b50a8cb180.PNG"></td>
  </tr></table></td></tr><tr><td><table><tr><td>Random flipping of image left/right to change the steering angle</td></tr><tr><td>
  <Image src="https://user-images.githubusercontent.com/25223180/53691051-68907180-3d9c-11e9-9706-8ac796de8427.PNG"></td></tr><tr>
  <td><Image src="https://user-images.githubusercontent.com/25223180/53691061-9675b600-3d9c-11e9-8c6e-8cfc17a5dbad.PNG"></td></tr><tr>
  <td><table><tr><td><Image src="https://user-images.githubusercontent.com/25223180/53691088-100da400-3d9d-11e9-893a-e37ae786b0ff.PNG">
  </td></tr><tr><td><Image src="https://user-images.githubusercontent.com/25223180/53691090-1439c180-3d9d-11e9-9f9a-c2ad427fd257.PNG">
  </td></tr></table></td></tr></table></td></tr><tr><td><table><tr><td>Randomly translate image within translation range of 100 and  
  modify steering angle accordingly to handle generalization</td><tr><tr><td>
  <Image src="https://user-images.githubusercontent.com/25223180/53694021-ad350080-3dce-11e9-890d-4858ba20b44b.PNG"></td></tr></table>
  </td></tr><tr><td><table>
  <tr><td>Random alteration of image brightness by converting the image from RGB  Color Space to HSV Color Space and scaling the value 
  of V-Channel with a random number in the range of (0.25 to 1.25) and then converting the image back to RGB Color Space from HSV Color 
  Space</td><tr><tr><td><table><tr><td>
  <Image src="https://user-images.githubusercontent.com/25223180/53691580-efe2e280-3da6-11e9-9ea3-1f2dc37f982a.PNG"></td></tr><tr>
  <td><Image src="https://user-images.githubusercontent.com/25223180/53691708-ab0c7b00-3da9-11e9-9728-ffa50194cd32.PNG"></td>
  </tr></table></td></tr><tr><td><table><tr><td>
  <Image src="https://user-images.githubusercontent.com/25223180/53691589-24ef3500-3da7-11e9-8558-aa2f490ece6e.PNG"></td></tr><tr>
  <td><Image src="https://user-images.githubusercontent.com/25223180/53691597-4e0fc580-3da7-11e9-8da7-966f9641d505.PNG"></td></tr>
  </table></td></tr><tr><td><table><tr><td>
  <Image src="https://user-images.githubusercontent.com/25223180/53691606-6da6ee00-3da7-11e9-8718-6c993ba487c6.PNG"></td></tr><tr>
  <td><Image src="https://user-images.githubusercontent.com/25223180/53691608-78fa1980-3da7-11e9-8251-9ba1aef56196.PNG"></td></tr>
  </table></td></tr></table></td></tr><tr><table><tr>
  <td>Converting Images from BGR Color Space to RGB Color Space</td></tr><tr><td>
  <Image src="https://user-images.githubusercontent.com/25223180/53690634-57436700-3d94-11e9-8b6e-dc7cfe42c685.PNG"></td></tr><tr>
  <td><Image src="https://user-images.githubusercontent.com/25223180/53690700-b2c22480-3d95-11e9-82d3-e762d3c614c3.PNG"></td></tr><tr>
  <td><Image src="https://user-images.githubusercontent.com/25223180/53690745-688d7300-3d96-11e9-8ed9-7ddef379a9d4.PNG"></td></tr>
  </table></td><td><table><tr><td><table><tr><td colspan=3 align="center">Random Selection of Left/Right or Center Images with STEERING
  ANGLE CORRECTION value as (+/-) 0.2</td></tr><tr><td> STEERING ANGLE CORRECTION</td><td>LEFT IMAGE</td><td>(+) 0.2</td></tr><tr> 
  <td colspan=3><Image src="https://user-images.githubusercontent.com/25223180/53692579-78b64a00-3db8-11e9-92ad-882779217b3b.PNG"></td>
  </tr><tr><td>STEERING ANGLE CORRECTION</td><td>RIGHT IMAGE</td><td>(-) 0.2</td></tr><tr>
  <td colspan=3><Image src="https://user-images.githubusercontent.com/25223180/53692575-7358ff80-3db8-11e9-9808-efa292d94255.PNG"></td>
  </tr></table></td></tr><tr><td><table><tr><td>Random flipping of image left/right to change the steering angle</td></tr><tr><td>
  <Image src="https://user-images.githubusercontent.com/25223180/53692739-1f034f00-3dbb-11e9-8ed9-e42ba143c479.PNG"></td>
  </tr><tr><td><Image src="https://user-images.githubusercontent.com/25223180/53692862-fd0acc00-3dbc-11e9-88fc-a60c924c5b10.PNG"></td>
  </tr><tr><td><Image src="https://user-images.githubusercontent.com/25223180/53692743-232f6c80-3dbb-11e9-9981-72744bb4b114.PNG"></td>
  </tr></table></td></tr><tr>
  <td><table><tr><td>Randomly translate image horizontally/vertically with steering angle adjustment to handle generalization</td></tr>
  <tr><td><Image src="https://user-images.githubusercontent.com/25223180/53692863-02681680-3dbd-11e9-84b4-4796c3d5daed.PNG"></td></tr>
  <tr><td><Image src="https://user-images.githubusercontent.com/25223180/53692865-072cca80-3dbd-11e9-9370-b7587ffe4999.PNG"></td></tr>
  </table></td></tr><tr><td><table><tr><td>Random  alteration of image brightness (lighter or darker)</td></tr><tr><td>
  <Image src="https://user-images.githubusercontent.com/25223180/53693857-38f95d80-3dcc-11e9-9f07-3b4693d5f640.PNG"></td></tr><tr>
  <td><Image src="https://user-images.githubusercontent.com/25223180/53693859-3d257b00-3dcc-11e9-8148-18c85def57d3.PNG"></td></tr><tr>
  <td><Image src="https://user-images.githubusercontent.com/25223180/53693862-4282c580-3dcc-11e9-8ba6-94d086fca7a1.PNG"></td></tr><tr>
  <td><Image src="https://user-images.githubusercontent.com/25223180/53693863-46164c80-3dcc-11e9-9295-9d80574c3fe7.PNG"></td></tr><tr>
  <td><Image src="https://user-images.githubusercontent.com/25223180/53693864-4adb0080-3dcc-11e9-9ed4-84c084299ce7.PNG"></td></tr>
  </table></td></tr><tr><td><table><tr><td>Random addition of Shadows to training Images Data</td></tr><tr><td>
  <Image src="https://user-images.githubusercontent.com/25223180/53693792-1b77c400-3dcb-11e9-9803-6f5286c3a849.PNG"></td></tr><tr>
  <td><Image src="https://user-images.githubusercontent.com/25223180/53693793-1fa3e180-3dcb-11e9-915e-b66400b16930.PNG"></td></tr><tr>
  <td><Image src="https://user-images.githubusercontent.com/25223180/53693797-25012c00-3dcb-11e9-8334-690190c8bb1b.PNG"></td></tr><tr>
  <td><Image src="https://user-images.githubusercontent.com/25223180/53693798-2c283a00-3dcb-11e9-9638-7d471a28251a.PNG"></td></tr><tr>
  <td><Image src="https://user-images.githubusercontent.com/25223180/53693800-34807500-3dcb-11e9-8588-d63e61229956.PNG"></td></tr>
  </table></td></tr></table></td></tr>
</table>

### Solution Design Approach
I decided to go for a supervised learning based image classification / regression algorithm implementation using deep convolutional
neural network after looking at the End-to-end solution provided by NVIDIA Team for a similar problem statement
I decided to use Keras library as it is a high-level wrapper over Tensorflow and Theano libraries and can be used for designing and
training a deep Convolutional-Net model using relatively less number of functions compared to Tensorflow.
Initially I implemented few traditional data-augmentation techniques with Version 1 Model and on testing found that the performanace 
of the model in predicting the steering angles could be improved more .
Hence while training the Version 2 model I took inspiration from the data augmentaion techniques suggested by the NVIDIA team 
like for instance converting the training image input into YUV Color Space from RGB color space along with the other data 
augmentation techniques used previously and training with a relatively larger training data set size and higher batch size with the 
value of sub-samples per epoch being set to 2000. Also I cropped the image beforehand to remove the top and bottom pixels containing
the sky and car front and resized the cropped image to the shape suggested by the NVIDIA state-of-the art solution for simialr problem 
statement.
Besides I used Dropout layers in both Version 1 model architecture (model.h5) and Version 2 model architecture (model_updated.h5)
to reduce overfitting on the training data-set using a probability value of 0.5 in both the versions of the CNN-Model.
These modifications along with few changes in the Version 1 ConvNet Model helped in achieving better control over prediction of the
steering angle of the simulated car when tested upon the lake track simulator enviroment in Unity than previosly achieved.
Stating that , however , nonetheless the Version 1 Model also performs as expected but the Version 2 model performs relatively better
in comparison.

### CONCLUSION
Using Data Augmentaion techniques and Conv-Net architecture model inspired by the NVIDIA End-to-End Self-Drive Car Implementation along 
with few changes in Deep CNN model architecture helped in achieving slightly better results compared to Version 1 Model in training and 
controlling the simulated car over Lake Track Simulation Environment.

### RESULTS SHOWCASE
<table>
  <tr>
    <td colspan="3" align="center">Lake Track</td>
  </tr>
  <tr>
    <td> </td>
    <td>Third Person View</td>
    <td>First Person View</td>
  </tr>
  <tr>
    <td>Version1</td>
    <td><table><tr><td></td><td><img src="https://user-images.githubusercontent.com/25223180/53681408-53b9cc80-3d0f-11e9-9dde-4bd8b667382d.png"></td></tr><tr><td>LINK</td>     <td>https://youtu.be/iysVoURBi-w</td></tr></table></td>
    <td><table><tr><td></td><td><img src="https://user-images.githubusercontent.com/25223180/53681614-012ddf80-3d12-11e9-82f4-1aa8abece6e3.png"></td></tr><tr><td>LINK</td><td>https://youtu.be/OOWMpntXZdo</td></tr></table></td>
  </tr>
  <tr>
    <td>Version2</td>
    <td><table><tr><td></td><td><img src="https://user-images.githubusercontent.com/25223180/53681524-fd4d8d80-3d10-11e9-88ff-49dc3e94af58.png"></td></tr><tr><td>LINK</td><td>https://youtu.be/lQPSs7eMMDc</td></tr></table></td>
    <td><table><tr><td></td><td><img src="https://user-images.githubusercontent.com/25223180/53681720-7d74f280-3d13-11e9-84ca-8198ebafb1a5.png"></td></tr><tr><td>LINK</td><td>https://youtu.be/boSpQ0HSPIQ</td></tr></table></td>
  </tr>
</table>

### FURTHER IMPROVEMENTS (FUTURE WORK SCOPE)
1) Collect Training Data by running the Car-Agent in Jungle Track Environment so that the Neural-Net model can learn to control
the car and predict accurate steering angle values to remain on the Jungle Track always in autonomous mode.

2) Using more data augmentation techniques to enhance and modify the training data set so that a generlaized scalable solution
can be achieved where the Simulated Car-Agent becomes capable of maneuvering itself properly on any kind of road/tracks
(i.e. Miscellaneous Simulation Environemnts) it is tested upon

### REFERENCES
Self-Driving Car Simulator (LINK : https://github.com/udacity/self-driving-car-sim )   
NVIDIA Deep ConvNet Model Architecture (LINK : https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/ ) 
   
