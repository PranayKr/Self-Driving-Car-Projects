# An Image Analysis Pipeline Implementation using the concepts of Gaussian Smoothing / Canny Edge Detection / Image Masking / Hough Line Transform algorithms to detect lane lines on road in LIVE-CAM Video Stream from a camera fixed on top of an autonomous car

# Problem Statement Decsription 

For this project the task is to detect lane lines on road from LIVE-Video Feed of a camera fixed on top of a moving autonomous car

# Results Showcase :

![lane-detector-self-drive-car-result1](https://user-images.githubusercontent.com/25223180/51797438-9af10100-2229-11e9-81c0-67c4c8f1ec13.gif)
# Result 1

![lane-detector-self-drive-car-result2](https://user-images.githubusercontent.com/25223180/51797582-1784df00-222c-11e9-9a2b-8905d35dc0e3.gif)
# Result 2 

Graphs of the Image (solidYellowCurve.jpg) showcasing the Pipeline Results at various stages

![image2_greyscale](https://user-images.githubusercontent.com/25223180/51797644-4cddfc80-222d-11e9-8793-19b23c551e98.PNG)

![image2_houghtrnsfrm](https://user-images.githubusercontent.com/25223180/51797648-5bc4af00-222d-11e9-848d-1163f3df8194.PNG)

![image2_finalresult](https://user-images.githubusercontent.com/25223180/51797650-697a3480-222d-11e9-8cdf-f5e02ce455b1.PNG)

# Result 3

# Installation Instructions to setup the Project :
# 1) Setting Up Python Environment :
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
    
  c) Clone the repository (https://github.com/udacity/CarND-LaneLines-P1)
     Then, install several dependencies (numpy/matplotlib/opencv) by executing the below commands in Anaconda Prompt Shell :
     
     Windows:
     1) conda install numpy
     2) conda install -c conda-forge matplotlib     
     3) conda install -c conda-forge opencv
     
  d) Create an Ipython Kernel for the sdcnd environment :
      
     python -m ipykernel install --user --name sdcnd --display-name "sdcnd"   
          
  e) Navigate to the CarND-LaneLines-P1/ folder 
  
     Before running code in the notebook (P1.ipynb), change the kernel to match the sdcnd environment by using the drop-down Kernel 
     menu.
     
# Details of running the Code Implementation :
  1) First of all clone this repository (https://github.com/PranayKr/Self-Driving-Car-Projects.git) on local system.
  2) Open Anaconda prompt shell window and navigate inside the CarND-LaneLines-P1/ folder in the Deep-RL cloned repository folder.
  3) Run the command "jupyter notebook" from the Anaconda prompt shell window to open the jupyter notebook web-app tool in the browser
     from where the source code present in notebook(.ipynb file) can be opened.
  4) Before running/executing code in a notebook, change the kernel (IPython Kernel created for sdcnd environment) to match the sdcnd
     environment by using the drop-down Kernel menu.
  5) The source code present in the provided notebook (.ipynb file) can also be collated in respective new python
     file (.py file) and then executed directly from the Anaconda prompt shell window using the command "python <filename.py>". 
  
  NOTE:
  All the cells can executed at once by choosing the option (Restart and Run All) in the Kernel Tab.
  
  Lane Lines Detection Algorithm Implementation Details (Files Used) :
  
  Open the mentioned Jupyter Notebook and execute all the cells : P1.ipynb
  
  Image Files used for Testing the Algorithm (present in the "test_images" folder):
  
  1)'solidWhiteCurve.jpg',
  2)'solidWhiteRight.jpg',
  3)'solidYellowCurve.jpg',
  4)'solidYellowCurve2.jpg',
  5)'solidYellowLeft.jpg',
  6)'whiteCarLaneSwitch.jpg'

  Video Files used for Testing the Algorithm (present in the "test_videos" folder):
  
  1)'solidWhiteRight.mp4'
  2)'solidYellowLeft.mp4'
  
  Results Output of Videos provided after Testing the Algorithm (present in the "test_videos_output" folder):
  
  1) 'solidWhiteRight_FiveSec.mp4'
  2) 'solidWhiteRight.mp4'
  3) 'solidYellowLeft.mp4'
  4) 'solidYellowLeft_FiveSec.mp4'
  
  Results Output of Images provided after Testing the Algorithm (present in the "test_images_output" folder):
  
  1)'solidWhiteCurve_OutputResult.PNG'
  2)'solidWhiteRight_OutputResult.PNG'
  3)'solidYellowCurve_OutputResult.PNG'
  4)'solidYellowCurve2_OutputResult.PNG'
  5)'solidYellowLeft_OutputResult.PNG'
  6)'whiteCarLaneSwitch_OutputResult.PNG'
  
  
  


  



