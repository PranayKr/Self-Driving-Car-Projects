# **Traffic Sign Recognition** 

The Software Pipeline built for Traffic Sign consists of the following steps:
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

#### An exploratory visualization of the dataset.

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

## Examples of Original Images 
![original_data](https://user-images.githubusercontent.com/25223180/53284432-834e5f00-377a-11e9-89aa-83e307b3911f.PNG)
## Examples of Preprocessed Images and Fake Data generated using Data Augmentation and Normalization
![augmented_data](https://user-images.githubusercontent.com/25223180/53284433-89444000-377a-11e9-8165-7d80879c12b2.PNG)
![augmented_data2](https://user-images.githubusercontent.com/25223180/53284435-8d705d80-377a-11e9-9a06-b774d31ac4d7.PNG)
![augmenetd_normalized_data](https://user-images.githubusercontent.com/25223180/53284438-9103e480-377a-11e9-8026-45bcd26346fc.PNG)
#### 2. Convolutional Neural Net Architecture 
#### Final model architecture:

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

#### HYPERPARAMETRS
1) Batch_Size = 64
2) NUMBER OF EPOCHS =50
3) LEARNING RATE = 1.0e-4
4) DROPOUT PROBABILITY = 0.5

#### SOLUTION APPROACH
### The first model architecture I used :
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

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


