# Implementation of an Advanced Lane-Detection-Algorithm by applying concepts of Camera Calibration , Distortion Correction , Color Transform , Gradient Threshold , Perspective Transform , Histogram Peak and Sliding Windows methods for Lane Detection , Radius of Curvature and Relative Vehicle Postion Calculations

# Problem Statement Description
For this project the task is to accurately identify lane boundaries in a video from a front-facing camera fixed on top of a car by 
coding an Image Analysis Pipeline using multiple image processing concepts and algorithms

# Results Showcase
![advanced-lane-detection-result](https://user-images.githubusercontent.com/25223180/52178715-e7bf7380-27f7-11e9-9aff-95dbe9aa0410.gif)
# Lane Detection Result on Video
![undistorted_checkboard14](https://user-images.githubusercontent.com/25223180/52179472-1beb6200-2801-11e9-8fcf-bbff1d1957d5.PNG)
# Distortion Correction on Chessboard Image
![undistorted_testimg1](https://user-images.githubusercontent.com/25223180/52179509-974d1380-2801-11e9-8a0c-922a1d7933a5.PNG)
# Distortion Correction on Test Image
![binary_threshimg_1](https://user-images.githubusercontent.com/25223180/52179538-edba5200-2801-11e9-817a-51ec359e4cc0.PNG)
# Binary Threshold Image
![perspectivetrnsfrm_img1](https://user-images.githubusercontent.com/25223180/52179550-0b87b700-2802-11e9-8dfa-352a90c8844b.PNG)
# Perspective Transformed Image
![lanes_detected_1](https://user-images.githubusercontent.com/25223180/52179561-29edb280-2802-11e9-9153-87d384eaa83f.PNG)
# Detected Lanes Plotted on Warped Image
![finalimageresult1](https://user-images.githubusercontent.com/25223180/52179579-4f7abc00-2802-11e9-9ded-bfcb8ed23e3f.PNG)
# Detected Lanes Plotted on Original Image
# Installation Instructions to setup the Project :
### Setting Up Python Environment :
  a) Download and install Anaconda 3 (latest version 5.3) from this link (https://www.anaconda.com/download/)
    for the specific Operating System and Architecure (64-bit or 32-bit) being used
    for Python 3.6 + version onwards
    
  b) Create (and activate) a new environment with Python 3.6.:
    Open Anaconda prompt and then execute the below given commands
    
    Linux or Mac:
    conda create --name sdcnd python=3.6
    source activate sdcnd
    
    Windows:
    conda create --name sdcnd python=3.6 
    activate sdcnd
    
  c) Clone the repository (https://github.com/udacity/CarND-Advanced-Lane-Lines.git)
     Then, install the dependencies (numpy/matplotlib/opencv and other libraries) by executing the below commands in Anaconda Prompt 
     Shell :
     
     Windows:
     1) conda install numpy
     2) conda install pandas
     3) conda install -c conda-forge matplotlib     
     4) conda install -c conda-forge opencv
     5) Install other libraries as required using conda package manager
     
  d) Create an Ipython Kernel for the sdcnd environment :
      
     python -m ipykernel install --user --name sdcnd --display-name "sdcnd"   
          
  e) Navigate to the CarND-Advanced-Lane-Lines/ folder 
  
     Before running code in the notebook (Advanced_Lane_Detection.ipynb), change the kernel to match the sdcnd environment by using the 
     drop-down Kernel menu.
     
     
# Details of running the Code Implementation :
  1) First of all clone this repository (https://github.com/PranayKr/Self-Driving-Car-Projects.git) on local system.
  2) Open Anaconda prompt shell window and navigate inside the CarND-Advanced-Lane-Lines/ cloned repository folder.
  3) Run the command "jupyter notebook" from the Anaconda prompt shell window to open the jupyter notebook web-app tool in the browser
     from where the source code present in notebook(.ipynb file) can be opened.
  4) Before running/executing code in a notebook, change the kernel (IPython Kernel created for sdcnd environment) to match the sdcnd
     environment by using the drop-down Kernel menu.
  5) The source code present in the provided notebook (.ipynb file) can also be collated in respective new python
     file (.py file) and then executed directly from the Anaconda prompt shell window using the command "python <filename.py>". 
  
  ### NOTE:
  All the cells can executed at once by choosing the option (Restart and Run All) in the Kernel Tab.
  
  ### Lane Lines Detection Algorithm Implementation Details (Files Used) :
  
  ### Open the mentioned Jupyter Notebook file and execute all the cells : Advanced_Lane_Detection.ipynb
  
  ### Image Files used for Testing the Algorithm (present in the "test_images" and "camera_cal" folder):
  1) 'calibration*.jpg' (* in range of 1 to 10)
  2) 'test*.jpg' (* in range of 1 to 6)
  3) 'straight_lines*.jpg' (* in range of 1 to 2)
 
  ### Video Files used for Testing the Algorithm (present in the "test_videos" folder):
  1) 'project_video.mp4'
  2) 'challenge_video.mp4'
  2) 'harder_challenge_video.mp4'
  
  ### Results Output of Videos provided after Testing the Algorithm (present in the "test_videos_output" folder):
  1) 'project_video_output.mp4'
  2) 'challenge_video_output.mp4'
  3) 'harder_challenge_video_output.mp4'
 
  ### Results Output of Images provided after Testing the Algorithm (present in the "Undistorted_Images", 
  ### "Binary_Threshold_Images","PerspectiveTrnsfrm_Images","Display_DetectedLanes" ,
  ### "Final_TestImagesResult" and "Final_TestImagesResult_Validated" folders):
  1) 'Undistorted_Checkboard*.PNG' (* in range of 1 to 20)
  2) 'Undistorted_TestImg*.PNG' (* in range of 1 to 8)
  3) 'Binary_ThreshImg_*.PNG' (* in range of 1 to 6)
  4) 'PerspectiveTrnsfrm_Img*.PNG' (* in range of 1 to 6)
  5) 'Lanes_Detected_*.PNG' (* in range of 1 to 6)
  6) 'FinalImageResult*.PNG' (* in range of 1 to 6)
  7) 'FinalImg_Validated*.PNG' (* in range of 1 to 6)
