# Implementation of Traffic-Sign-Classification using Convolutional Neural Network Model built using Tensorflow Library and Image preprocessing Techniques of Data Augmentation and Data Normalization and German Traffic Sign Dataset 

# Problem Statement Description
For this project the task is to build a Convolutional Neural-Net Model which can classify 43 different categories of Traffic Signs

# Training Data ShowCase 
![train_data_1](https://user-images.githubusercontent.com/25223180/53285884-7175b780-378c-11e9-90d3-aee741ce206b.PNG)
![train_data_2](https://user-images.githubusercontent.com/25223180/53285888-78042f00-378c-11e9-978e-d10cb319d0c4.PNG)
# Results ShowCase
![image1](https://user-images.githubusercontent.com/25223180/53285705-bba96980-3789-11e9-8add-7f1b2e19d1ed.PNG)
![image1_pred](https://user-images.githubusercontent.com/25223180/53285709-c237e100-3789-11e9-9aca-51687340a055.PNG)
![image2](https://user-images.githubusercontent.com/25223180/53285714-cf54d000-3789-11e9-94c2-f2f35d3b5c87.PNG)
![image2_pred](https://user-images.githubusercontent.com/25223180/53285718-d976ce80-3789-11e9-8978-d0fd1e967a49.PNG)
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
    
  c) Download the German Traffic Signs Dataset from this link
     https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip
    
  d) Clone the repository (https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project)
     Then, install the dependencies (numpy/matplotlib/opencv and other libraries) by executing the below commands in Anaconda Prompt 
     Shell :
     
     Windows:
     1) conda install numpy
     2) conda install pandas
     3) conda install -c conda-forge matplotlib     
     4) conda install -c conda-forge opencv
     5) Install other libraries as required using conda package manager
     
  e) Create an Ipython Kernel for the sdcnd environment :
      
     python -m ipykernel install --user --name sdcnd --display-name "sdcnd"   
          
  f) Navigate to the CarND-Traffic-Sign-Classifier-Project/ folder 
  
     Before running code in the notebook (Traffic_Sign_Classifier-Implementation.ipynb), change the kernel to match the sdcnd
     environment by using the  drop-down Kernel menu.
         
# Details of running the Code Implementation :
  1) First of all clone this repository (https://github.com/PranayKr/Self-Driving-Car-Projects.git) on local system.
  2) Open Anaconda prompt shell window and navigate inside the CarND-Traffic-Sign-Classifier-Project/ cloned repository folder.
  3) Run the command "jupyter notebook" from the Anaconda prompt shell window to open the jupyter notebook web-app tool in the browser
     from where the source code present in notebook(.ipynb file) can be opened.
  4) Before running/executing code in a notebook, change the kernel (IPython Kernel created for sdcnd environment) to match the sdcnd
     environment by using the drop-down Kernel menu.
  5) The source code present in the provided notebook (.ipynb file) can also be collated in respective new python
     file (.py file) and then executed directly from the Anaconda prompt shell window using the command "python <filename.py>". 
  
  ### NOTE:
  All the cells can executed at once by choosing the option (Restart and Run All) in the Kernel Tab.
  
  ### Traffic Sign Classification Algorithm Implementation Details (Files Used) :
  
  ### Open the mentioned Jupyter Notebook file and execute all the cells : Traffic_Sign_Classifier-Implementation.ipynb
  
  ### Training/ Validation/ Test Data Pickled files provided in the "traffic-signs-data" folder:
  1) test.p
  2) train.p
  3) valid.p
 
  ### Image Files used for Testing the Algorithm (present in the "trafficsigns_german\RandomImages" folder):
  1) 'traffic_sign_1.png' 
  2) 'traffic_sign_2.jpg' 
  3) 'traffic_sign_3.png'
  4) 'traffic_sign_4.jpg'
  5) 'traffic_sign_5.jpg' 
  
  ### Excel File provided having details of different Traffic Sign types and their corresponding ids: 
  signnames.csv
  
  ### Pretrained Model files provided in the "models" folder :
  1) trafficlight_classifier_model.data-00000-of-00001
  2) trafficlight_classifier_model.index
  3) trafficlight_classifier_model.meta
      
      
