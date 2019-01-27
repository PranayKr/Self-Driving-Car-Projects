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
  
     Before running code in a notebook, change the kernel to match the sdcnd environment by using the drop-down Kernel menu.
     




