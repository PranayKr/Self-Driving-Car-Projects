# Implementation of Car-Driving Behavior Cloning using Unity Self-Driving Car Simulator and Convolutional Neural Network Model with Keras Library to execute Supervised Learning with Logistic Regression Algorithm to predict correct steering angle to be taken by the simulated car based on the inputs of LIVE Feed of Center , Left and Right Cameras mounted on top of the Car 

# Problem Statement Description
For this project the task is to accurately predict the steering angle to be taken by the simulated car based on the inputs of LIVE Camera Feed of Center , Left and Right Cameras mounted on top of the Car and hence clone / imitate the driving behaviour of a real person behind the steering wheel.

# Results Showcase
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

# Installation Instructions to setup the Project :
### Setting Up Python Environment :
  a) Download and install Anaconda 3 (latest version 5.3) from this link (https://www.anaconda.com/download/)
     for the specific Operating System and Architecure (64-bit or 32-bit) being used
     for Python 3.6 + version onwards
     
  b) Download and install Game Engine Software Tool Unity (https://unity3d.com/) 
    
  c) Clone the mentioned Repository using Anaconda Prompt Shell
     ([CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit))
     
     The lab enviroment can be created with CarND Term1 Starter Kit. 
     Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.
          
  d) Clone the Self-Driving Car Simulator Repository
     (https://github.com/udacity/self-driving-car-sim.git)
     
  e) Load the self-drive-car-sim project in Unity IDE  
   
# Details of Training the model
  1) Start up the Udacity self-driving car simulator in Unity , choose a scene and press the Training Mode button.
  2) Select a folder in which all the training images generated during manual run would be saved
  3) Maneuver the Simulated Car in the Unity Environment in either Lake Road environment or Jungle Track environment
     to generate training images
  4) After the training images have been generated run either of the following commands in Ananconda Prompt Shell
     
     a) Version 1
        ```sh
        python model.py
        ```
     b) Version 2
        ```sh
        python model_V2.py
        ```
     After Training is over a file "model.h5" would be generated for Version 1 or a file "model_updated.h5" would be generated for
     Version 2

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
# Behavior Cloning Implementation Details (Files Used) :

### Model Architecture / Image Data Preprocessing and Model Training Logic Files:
  1) model.py (Version 1) 
  2) model_V2.py (Version 2)

### Excel File having the log data mapping the Training Data Images to corresponding values of steering,throttle,break and speed: 
  driving_log.csv
  
### Pretrained Model files provided in the "models" folder :
  1) model.h5 (Version 1) 
  2) model_updated.h5 (Version 2) 
  
### Python Script to drive the car in autonomous mode in the Unity Environment Simulator:
  1) drive.py (Version 1) 
  2) drive_V2.py (Version 2)
  
### Python Script to generate video from the iamge frames captured from the last run in the Simulator:
  video.py
  
### Output Results Videos Files :
  1) sim_run_V1.mp4 (Version 1) (First Person View) (LINK : https://youtu.be/iysVoURBi-w )
  2) sim_run_v2.mp4 (Version 2) (First Person View) (LINK : https://youtu.be/lQPSs7eMMDc )
  3) Version 1 (Third Person View) (LINK : https://youtu.be/OOWMpntXZdo )
  4) Version 2 (Third Person View) (LINK : https://youtu.be/boSpQ0HSPIQ )
