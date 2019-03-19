# Implementation of Extended Kalman Filter in C++ using Unity Extended-Kalman-Filter Simulator , Simulated lidar and radar measurements readings detecting a bicycle travelling around the car in the simulation environment to track the bicycle's position and velocity.

# Problem Statement Description
For this project the task is to implement Extended Kalman Filter in C++ using Unity Extended-Kalman-Filter Simulator , simulated lidar and radar measurements readings detecting a bicycle travelling around the simulated car in the simulation environment to track the bicycle's position and velocity. Lidar measurements are red circles, radar measurements are blue circles with an arrow pointing in the direction of the observed angle, and estimation markers are green triangles.The simulator provides the script the measured data (either lidar or radar), and the script feeds back the measured estimation marker, and RMSE values from its Kalman filter.

# Solution Criteria
px, py, vx and vy RMSE (Root-Mean-Square-Error) values should be less than or equal to the values [.11, .11, 0.52, 0.52]
respectively.

# Results Showcase
<table>
  <tr>
    <td colspan="3" align="center">RESULTS</td>
  </tr>
  <tr>
    <td> </td>
    <td>Zoomed In View</td>
    <td>Zoomed Out View</td>
  </tr>
  <tr>
    <td>Dataset 1</td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54604926-027b3e00-4a6e-11e9-92a6-974434942301.gif"></td></tr><tr>
      <td>LINK</td>     
      <td>https://youtu.be/LJ9kPpbqWFk</td></tr></table></td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54604913-fa230300-4a6d-11e9-8ece-a657faaf7c27.gif"></td></tr><tr>
      <td>LINK</td><td>https://youtu.be/0U46mhdeEpg</td></tr></table></td>
  </tr>
  <tr>
    <td>Dataset 2</td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54604956-0dce6980-4a6e-11e9-9d34-c442ab505fd2.gif"></td></tr><tr>
      <td>LINK</td><td>https://youtu.be/LhkD237zxQQ</td></tr></table></td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54604944-07d88880-4a6e-11e9-90f4-f3b854858209.gif"></td></tr><tr>
      <td>LINK</td><td>https://youtu.be/1hsIQAJ0aRY</td></tr></table></td>
  </tr>
</table>

<table>
  <tr>
    <td colspan="3" align="center">FINAL OUTPUT SCREENSSHOT</td>
  </tr>
  <tr>
    <td> </td>
    <td>Zoomed In View</td>
    <td>Zoomed Out View</td>
  </tr>
  <tr>
    <td>Dataset 1</td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54605688-d9f44380-4a6f-11e9-8f12-45ac17ada9b2.png"></td></tr></table>
      </td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54605706-de206100-4a6f-11e9-818d-fbb33574dfb5.png"></td></tr></table></td>
  </tr>
  <tr>
    <td>Dataset 2</td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54605713-e1b3e800-4a6f-11e9-94a0-7ea8ef0f631d.png"></td></tr>
      </table></td>
    <td><table><tr><td></td><td>
      <img src="https://user-images.githubusercontent.com/25223180/54605718-e6789c00-4a6f-11e9-84ba-7e7778b3a8fe.png"></td></tr>
      </table></td>
  </tr>
</table>

# Installation Instructions to setup the Project :

1) Download the Unity Extended Kalman Filter Simulator Enviroment  
   from this link (https://github.com/udacity/self-driving-car-sim/releases)

2) Install the Ubuntu Bash Environment if using Windows 10 OS by following steps in the following link 
   (https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/)
   
   The Ubuntu Bash Environment will be used to install and run uWebSocketIO Package (https://github.com/uNetworking/uWebSockets). 
   This package facilitates the connection between the Unity Extended Kalman Filter Simulator Enviroment and the C++ Script Code 
   implementing the Extended Kalman Filter.The package does this by setting up a web socket server connection from the C++ program
   to the simulator, which acts as the host

### 3) Other Important Dependencies :
       a) cmake >= 3.5
          All OSes: Installation Instructions are in this link (https://cmake.org/install/)
       b) make >= 4.1 (Linux, Mac), 3.81 (Windows)
          1) Linux: make is installed by default on most Linux distros
          2) Mac: Installation Instructions are in this link (https://developer.apple.com/xcode/features/)
          3) Windows: Installation Instructions are in this link (http://gnuwin32.sourceforge.net/packages/make.htm)
       c) gcc/g++ >= 5.4
          1) Linux: gcc / g++ is installed by default on most Linux distros
          2) Mac: same deal as make (Installation Instructions are in this link (https://developer.apple.com/xcode/features/))
          3) Windows: recommend using MinGW (please follow this link to get details about MinGW (http://www.mingw.org/))
### 4) Launch the Ubuntu Bash Environment on system (using Windows 10) 
       After Setting the Username and Password execute the below mentioned commands :
       a) sudo apt-get update
       b) sudo apt-get install git
       c) sudo apt-get install cmake
       d) Installing cmake requires g++ compiler. Install a g++ version 4.9 or greater. 
          Execute the following commands to install g++ compiler:
          1) sudo add-apt-repository ppa:ubuntu-toolchain-r/test
          2) sudo apt-get update
          3) sudo apt-get install g++-4.9
       e) sudo apt-get install openssl
       f) sudo apt-get install libssl-dev
       g) Clone the repository (https://github.com/udacity/CarND-Kidnapped-Vehicle-Project) using the following command:
          git clone https://github.com/udacity/CarND-Kidnapped-Vehicle-Project
       h) sudo rm /usr/lib/libuWS.so
       i) navigate to CarND-Kidnapped-Vehicle-Project/ folder in the Ubuntu Bash Environment and execute the following command:
          ./install-ubuntu.sh
   ## BUILD Instructions to Compile the Source Code files:
      j) a) mkdir build (create a build directory)
         b) cd build ( go inside the /build directory)
         c) cmake ..
         d) make
   ## LAUNCH THE SIMULATOR
      k) a) Open the Dowloaded Simulator (link :https://github.com/udacity/self-driving-car-sim/releases) in Windows 10 environment
         b) In the main menu screen select Project 1/2 EKF and UKF
         c) Once the scene is loaded hit the START button to observe how the object moves and how measurement markers are positioned in 
            the data set."Data set 2" is also included which is a reversed version of "Data set 1".The "Data set 2" starts with a radar 
            measurement where the "Data set 1" starts with a lidar measurement. PAUSE button can be pressed to pause the scene or the 
            RESTART button can be used to reset the scene. The ARROW KEYS can be used to move the camera around, and the top left ZOOM 
            IN/OUT buttons can be used to focus the camera. Pressing the ESCAPE KEY returns to the simulator main menu.
         d) Hitting Restart or switching between data sets only refreshes the simulator state and not the Kalman Filter's saved results. 
   ## RUN the Source Code
      l) Navigate to /build directory and execute the following command after the Simulator has been launched and project environment 
         loaded:
          ./ExtendedKF
       
         On receiving the following message "Listening to port 4567 Connected!!!" is a confirmation that the C++ Script
         is ready to communicate with the Simulator Environemnt using the C++ uWebSocketIO Package.
       
   ## Transfer of files between the Windows 10 Environment and Ubuntu Bash Environment
       1) Download and install Anaconda 3 (latest version 5.3) from this link (https://www.anaconda.com/download/) for the specific 
          Operating System and Architecure (64-bit or 32-bit) being used for Python 3.6 + version onwards
       2) Open Anaconda Prompt Shell Window
       3) Clone the repository (https://github.com/udacity/CarND-Kidnapped-Vehicle-Project) using the following command:
          git clone https://github.com/udacity/CarND-Kidnapped-Vehicle-Project
       4) Edit the C++ Source Files and C++ Header Files as required using some text editor like VS Code IDE Tool
       5) Mount the drive (where the repository has been cloned in Windows 10 environment) in Ubuntu BASH using the following command:
          cd /mnt /e 
       6) Navigate to the src/ folder inside CarND-Kidnapped-Vehicle-Project/ folder present in the e drive (in this case) 
          using the following command (where all the modified source code files are present):
          cd CarND-Extended-Kalman-Filter-Project/src
       7) Copy the modified files to the desired location in Ubuntu BASH Environment , navigated to the appropriate Ubuntu BASH folder
          using the following command :
          cp * <appropriate Ubuntu BASH folder path>
       
   ## Generation of Additional LIDAR and RADAR Simulated Data
      1) Clone the repository (https://github.com/udacity/CarND-Mercedes-SF-Utilities) using the following command :
         git clone https://github.com/udacity/CarND-Mercedes-SF-Utilities
      2) Check the MATLAB Scripts provided inside the matlab_examples/ folder in the cloned repo to generate additional data
      3) Visualization Package also available as python scripts inside the python/ folder in the cloned repo
       
   ## SOURCE CODE FILE Functional Overview :
      1) Data File (obj_pose-laser-radar-synthetic-input.txt)
         The simulator will be using this data file, and feed main.cpp values from it one line at a time.
         Each row represents a sensor measurement where the first column tells you if the measurement comes from radar (R) or lidar (L). 
                      
         a) RADAR DATA STRUCTURE
            For a row containing radar data, the columns are: sensor_type, rho_measured, phi_measured, rhodot_measured, timestamp, 
            x_groundtruth, y_groundtruth, vx_groundtruth, vy_groundtruth, yaw_groundtruth, yawrate_groundtruth.
           
         b) LIDAR DATA STRUCTURE 
            For a row containing lidar data, the columns are: sensor_type, x_measured, y_measured, timestamp, x_groundtruth, 
            y_groundtruth, vx_groundtruth, vy_groundtruth, yaw_groundtruth, yawrate_groundtruth.
              
         Whereas radar has three measurements (rho, phi, rhodot), lidar has two measurements (x, y).
           
         Groundtruth, which represents the actual path the bicycle took, is for calculating root mean squared error.
         The code reads in the data file line by line. The measurement data for each line gets pushed onto a measurement_pack_list
         The ground truth [px,py,vx,vy] for each line in the data file gets pushed ontoground_truthso RMSE can be calculated later 
         from tools.cpp.
           
   ### KALMAN FILTER LOGIC PIPELINE EXPLANATION
       The three main steps for programming a Kalman filter are:
       1) initializing Kalman filter variables
       2) predicting where our object is going to be after a time step \Delta{t}Δt
       3) updating where our object is based on sensor measurements
       4) Then the prediction and update steps repeat themselves in a loop.
          
       To measure the accuracy of Kalman filter logic , root mean squared error is calculated comparing the Kalman filter results 
       with the provided ground truth values.
          
       These three steps (initialize, predict, update) plus calculating RMSE encapsulate the entire extended Kalman filter project.
          
      2) main.cpp : 
         a) communicates with the Unity Extended Kalman Filter Simulator Environment receiving data measurements from the data file
            obj_pose-laser-radar-synthetic-input.txt
         b) calls a function to run the Kalman filter
         c) calls a function to calculate RMSE (Root-Mean-Square-Error) Values
           
      3) FusionEKF.cpp : 
         a) initializes the Kalman Filter / Extended Kalman Filter class
         b) calls the predict function
         c) calls the update function
          
      4) kalman_filter.cpp : 
         a) defines the predict function, the update function for lidar, and the update function for radar
           
      5) tools.cpp :
         a) functions to calculate RMSE and the Jacobian matrix

   ### CODE FUNCTIONAL FLOW OVERVIEW:
       1) Main.cpp reads in the data and sends a sensor measurement to FusionEKF.cpp
       2) FusionEKF.cpp takes the sensor data and initializes variables and updates variables. 
          The Kalman filter equations are not in this file. FusionEKF.cpp has a variable called ekf_, which is an instance of a 
          KalmanFilter class. The ekf_ will hold the matrix and vector values.The ekf_ instance will be also used to call the predict 
          and update equations.
       3) The KalmanFilter class is defined in kalman_filter.cpp C++ source file and kalman_filter.h header file.
         
# Extended Kalman Filter Implementation Details (Files Used) :      
     
