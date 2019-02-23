# **Traffic Sign Recognition** 
# Problem Statement Description
For this project the task is to build a Convolutional Neural-Net Model which can classify 43 different categories of Traffic Signs

The Software Pipeline built for Traffic Sign Classification consists of the following steps:
* Load the German Traffic Sign Data Set (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Data Set Summary & Exploration
I used the pandas library to calculate summary statistics of the traffic
signs data set:
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32*32*3
* The number of unique classes/labels in the data set is 43

### An exploratory visualization of the dataset.
Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of the number of images for each 
traffic sign category
![traffic_sign_classes_distibution_plot](https://user-images.githubusercontent.com/25223180/53284275-952f0280-3778-11e9-9650-e880b3e3f724.PNG)
### Designing and Testing of Model Architecture
### Data Preprocessing and Data Augmentation Steps 
For Preprocessing and Augmenting the data I used the below mentioned image processing techniques:
1) Randomly adjusting brightness of the image.
2) Randomly rotating the image
3) Randomly moving/ warping the image
4) Randomly distorting the image
5) Normalizing the image ((train_img - train_img.mean())/train_img.std())

Data Normalization and Data Augmentation techniques were used to improve the learning scope of the CNN model so that the model can be
generalized and scaled so as to properly predict / classify any traffic sign image which might be completely different from the 
images the CNN model has been pretrained on .
This is because often in real-time scenarios the lighting conditions / angle of the camera / resolution of the camera would be varying
and a robust Image Classifier model should be able to correctly predict the traffic sign type even in such not-so ideal conditions

### Examples of Original Images 
![original_data](https://user-images.githubusercontent.com/25223180/53284432-834e5f00-377a-11e9-89aa-83e307b3911f.PNG)
### Examples of Preprocessed Images and Fake Data generated using Data Augmentation and Normalization
![augmented_data](https://user-images.githubusercontent.com/25223180/53284433-89444000-377a-11e9-8165-7d80879c12b2.PNG)
![augmented_data2](https://user-images.githubusercontent.com/25223180/53284435-8d705d80-377a-11e9-9a06-b774d31ac4d7.PNG)
![augmenetd_normalized_data](https://user-images.githubusercontent.com/25223180/53284438-9103e480-377a-11e9-8026-45bcd26346fc.PNG)
### Convolutional Neural Net Architecture 
### Final model architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image| 
| Convolution 5x5   | 1x1 stride, VALID padding, outputs 28x28x24|
| Max pooling	      | 2x2 stride,  outputs 14x14x24 				|
| RELU              |                                       |
| Convolution 5x5	  | 1x1 stride, VALID padding, outputs 10x10x64   |
| Max pooling	      | 2x2 stride,  outputs 5x5x64 				|
| RELU     |                                       |
| Dropout  |        | keep_probability =0.5        |
| Flatten  |        | Input = 5x5x64. Output = 1600|
| Fully connected		| Input = 1600. Output = 480   |
| RELU     |                                       |
| Fully connected		| Input = 480. Output = 43     |

### HYPERPARAMETERS
1) Batch_Size = 64
2) NUMBER OF EPOCHS =50
3) LEARNING RATE = 1.0e-4
4) DROPOUT PROBABILITY = 0.5

### SOLUTION APPROACH
### The first model architecture I had used :
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5   | 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU              |                                     |
| Max pooling	      | 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	  | 1x1 stride, VALID padding, outputs 10x10x16   |
| RELU              |                                     |
| Max pooling	      | 2x2 stride,  outputs 5x5x16 				|
| Flatten  |        | Input = 5x5x16. Output = 400        |
| Fully connected		| Input = 400. Output = 120   |
| RELU              |                             |
| Fully connected		| Input = 120. Output = 84    |
| RELU              |                             |
| Fully connected		| Input = 84. Output = 43    |


The first CNN model I added one more fully connected layer and also the number of output neurons for the convnet layers was less

I increased the number of output neurons of the convent layer so that more features of the training data are learned quickly 
during training phase.

The validation accuracy did not reach beyond the benchamark 93 % even after training for 50 epochs with the first architecture

Hence I made the changes mentioned above in the final CNN architecture.

Dropout was one more addition I made to the final architecture with probability set to 0.5 to tackle overfitting during training

I used a batch size of 64 so that learning of the model happnens quicker in a comparatively less number of epochs of 50 

I also lowered the learning rate from .001 to 1.0e-4 for stabilizing the learning of the model

Finally I achieved the following results :
1) Validation Accuracy = 0.936
2) Test Accuracy = 0.925
3) Accuracy on Random Images = 0.6 (60%)

### Prediction Results on Random New Images
### Random Images Used
![random_images](https://user-images.githubusercontent.com/25223180/53285415-2fe20e00-3786-11e9-8840-960226f1b38c.PNG)
The first image might be difficult to classify because of low lighting conditions in the image , whereas the second image is a
bit warped in prespective .

Taking into consideration all these miscellaneous conditions in which the model would have to correctly classify a traffic sign 
data augmnentation and normalization techniques were used 

### Model's predictions on these new traffic signs 
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      		| Right-of-way at the next intersection  (Correct)									| 
| Speed limit (60km/h)     			| Speed limit (60km/h) 	(Correct)						  |
| Speed limit (70km/h)					| General caution	 (Wrong)										|
| No passing	      		        | End of no passing   (Not Exactly)                |
| No passing by vehicles over 3.5 metric tons			| End of no passing by vehicles over 3.5 metric tons  (Not Exactly)    							|

The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. 

The last 2 misclassifications are very close as the model was correctly able to infer that the images are related to No Passing 
and No Passing for heavier vehicles

### The top 5 softmax probabilities for each image along with the sign type of each probability. 
![image1](https://user-images.githubusercontent.com/25223180/53285705-bba96980-3789-11e9-8add-7f1b2e19d1ed.PNG)
![image1_pred](https://user-images.githubusercontent.com/25223180/53285709-c237e100-3789-11e9-9aca-51687340a055.PNG)
![image2](https://user-images.githubusercontent.com/25223180/53285714-cf54d000-3789-11e9-94c2-f2f35d3b5c87.PNG)
![image2_pred](https://user-images.githubusercontent.com/25223180/53285718-d976ce80-3789-11e9-8978-d0fd1e967a49.PNG)
![image3](https://user-images.githubusercontent.com/25223180/53285719-ded41900-3789-11e9-87a5-0cc7455b0a87.PNG)
![image3_pred](https://user-images.githubusercontent.com/25223180/53285721-e398cd00-3789-11e9-9573-951f4050d9e2.PNG)
![image4](https://user-images.githubusercontent.com/25223180/53285722-e85d8100-3789-11e9-8757-b46a5ce8636a.PNG)
![image4_pred](https://user-images.githubusercontent.com/25223180/53285725-eeebf880-3789-11e9-93bd-8cde3506a018.PNG)
![image5](https://user-images.githubusercontent.com/25223180/53285728-f4e1d980-3789-11e9-8702-46a54c9debb9.PNG)
![image5_pred](https://user-images.githubusercontent.com/25223180/53285730-f8756080-3789-11e9-9648-05ea638278cd.PNG)

### POSSIBLE IMPROVEMENTS (FUTURE WORK)
### 1) Training the model for more number of Epochs more getting a more accurate well-learned model
### 2) Visualizing the Neural Network : Discussion and anaylysis the visual output of trained network's feature maps. 


