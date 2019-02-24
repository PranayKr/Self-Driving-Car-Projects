#!/usr/bin/env python
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[1]:


# Load pickled data
import pickle
import os
import cv2
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from random import randint
# Visualizations will be shown in the notebook.
get_ipython().run_line_magic('matplotlib', 'inline')

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
validation_file=  'traffic-signs-data/valid.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[2]:


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(pd.Series(y_train).unique())

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Number of validation examples =", n_validation)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# In[3]:


def plot_class_distribution(y_train, title="Training Data Distribution"):
    """
    Plot the traffic sign class distribution
    """
    plt.figure(figsize=(15, 5))
    plt.hist(y_train, bins=n_classes)
    plt.title(title)
    plt.xlabel('Sign')
    plt.ylabel('Count')
    plt.show()


# In[4]:


plot_class_distribution(y_train)


# In[5]:


sign_names_list = pd.read_csv('signnames.csv', index_col='ClassId')
print(sign_names_list)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[6]:


def random_brightness(image, ratio):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = np.float64(hsv[:, :, 2])
    brightness = brightness * (1.0 + np.random.uniform(-ratio, ratio))
    brightness[brightness>255] = 255
    brightness[brightness<0] = 0
    hsv[:, :, 2] = brightness
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def random_rotation(image, angle):
    """
    Randomly rotate the image
    """
    if angle == 0:
        return image
    angle = np.random.uniform(-angle, angle)
    rows, cols = image.shape[:2]
    size = cols, rows
    center = cols/2, rows/2
    scale = 1.0
    rotation = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, rotation, size)


def random_translation(image, translation):
    """
    Randomly move the image
    """
    if translation == 0:
        return 0
    rows, cols = image.shape[:2]
    size = cols, rows
    x = np.random.uniform(-translation, translation)
    y = np.random.uniform(-translation, translation)
    trans = np.float32([[1,0,x],[0,1,y]])
    return cv2.warpAffine(image, trans, size)


def random_shear(image, shear):
    """
    Randomly distort the image
    """
    if shear == 0:
        return image
    rows, cols = image.shape[:2]
    size = cols, rows
    left, right, top, bottom = shear, cols - shear, shear, rows - shear
    dx = np.random.uniform(-shear, shear)
    dy = np.random.uniform(-shear, shear)
    p1 = np.float32([[left   , top],[right   , top   ],[left, bottom]])
    p2 = np.float32([[left+dx, top],[right+dx, top+dy],[left, bottom+dy]])
    move = cv2.getAffineTransform(p1,p2)
    return cv2.warpAffine(image, move, size)
    
    
def augment_image(image, brightness, angle, translation, shear):
    image = random_brightness(image, brightness)
    image = random_rotation(image, angle)
    image = random_translation(image, translation)
    image = random_shear(image, shear)
    return image


# In[7]:


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.

#Load the signname csv file
df = pd.read_csv('signnames.csv', names=['TrafficSign_Id', 'Name'], header=0)


# In[8]:


def load_image(image_file):
    """
    Read image file into numpy array (RGB)
    """
    return plt.imread(image_file)


# In[9]:


def get_samples(image_data,X_train,y_train,num_train_images, num_samples):
          
    sign_image_list =[]
    image_name_list =[]
        
    for i in range(0,num_samples):
        index = randint(0,num_train_images)
        image = X_train[index]
        sign_image_list.append(image)
        image_name_list.append(image_data.Name[y_train[index]])
  
    return sign_image_list,image_name_list


# In[10]:


def show_images(image_data, cols=5, sign_names=None, show_shape=False, augment_Flag=False,NormalizeFlag=False):
    """
    load sample images and show them.
    """
    num_images = len(image_data)
    rows = num_images//cols
    plt.figure(figsize=(cols*3,rows*2.5))
    
    for counter, image in enumerate(image_data):    
        if augment_Flag is not False:
            image = augment_image(image,brightness=0.7, angle=10, translation=5, shear=2)
        if NormalizeFlag is not False:
            image=(image-image.mean())/image.std()
            
        plt.subplot(rows, cols, counter+1)
        plt.imshow(image)
        if sign_names is not None:
            plt.text(0, 0, '{}: {}'.format(counter, sign_names[counter]), color='k',backgroundcolor='y', fontsize=8)        
        if show_shape:
            plt.text(0, image.shape[0], '{}'.format(image.shape), color='k',backgroundcolor='c', fontsize=8)        
        plt.xticks([])
        plt.yticks([])
    plt.show()


# In[11]:


sample_data,sample_data_labels = get_samples(df,X_train,y_train,len(X_train), 40)

show_images(sample_data, sign_names=sample_data_labels, show_shape=True,augment_Flag=False,NormalizeFlag=False)


# ### Original Sample Data 

# In[12]:


show_images(sample_data[10:], cols=10)


# ### Generated Training Data using Data Augmentation

# In[13]:


for i in range(5):
    show_images(sample_data[10:], cols=10, augment_Flag=True,NormalizeFlag=False) # after data augmentation


# ### Generated Training Data using Data Augmentation and Normalization

# In[14]:


show_images(sample_data[10:], cols=10, augment_Flag=True,NormalizeFlag=True) # after data augmentation and normalization


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[16]:


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)
X_test, y_test = shuffle(X_test, y_test)

#image original
image = X_train[13]
plt.subplot(2,2,1)
plt.imshow(image)
plt.title(y_train[2])
plt.axis('off')


# ### Normalisation approach 1 ((X-X.mean())/(np.max(X)-np.min(X)))

# In[19]:


X_train_normalized = (X_train-X_train.mean())/(np.max(X_train)-np.min(X_train))
X_valid_normalized = (X_valid-X_valid.mean())/(np.max(X_valid)-np.min(X_valid))
X_test_normalized = (X_test-X_test.mean())/(np.max(X_test)-np.min(X_test))

#image modified
image = X_train_normalized[13]
plt.subplot(2,2,1)
plt.imshow(image)
plt.title(y_train[2])
plt.axis('off')


# ### Normalisation approach 2 (X-X.mean())/X.std()

# In[20]:


X_train_normalized = (X_train-X_train.mean())/X_train.std()
X_valid_normalized = (X_valid-X_valid.mean())/X_valid.std()
X_test_normalized = (X_test-X_test.mean())/X_test.std()

#image modified
image = X_train_normalized[13]
plt.subplot(2,2,1)
plt.imshow(image)
plt.title(y_train[2])
plt.axis('off')


# ### Model Architecture

# In[21]:


### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x): 
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = 0, stddev = 0.1))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    
    # Activation 1.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    
    # Layer 2: Convolutional. Input = 14x14x6. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = 0, stddev = 0.1))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation 2.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Flatten. Input = 5x5x16. Output = 400.
    flattened   = flatten(conv2)
    
    #Matrix multiplication
    #input: 1x400
    #weight: 400x120 
    #Matrix multiplication(dot product rule)
    #output = 1x400 * 400*120 => 1x120
    
     # Layer 3: Fully Connected. Input = 400. Output = 120.
    fullyc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = 0, stddev = 0.1))
    fullyc1_b = tf.Variable(tf.zeros(120))
    fullyc1   = tf.matmul(flattened, fullyc1_W) + fullyc1_b
    
    # Full connected layer activation 1.
    fullyc1    = tf.nn.relu(fullyc1)
    
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fullyc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = 0, stddev = 0.1))
    fullyc2_b  = tf.Variable(tf.zeros(84))
    fullyc2    = tf.matmul(fullyc1, fullyc2_W) + fullyc2_b
    
    # Full connected layer activation 2.
    fullyc2    = tf.nn.relu(fullyc2)
    
    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fullyc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = 0, stddev = 0.1))
    fullyc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fullyc2, fullyc3_W) + fullyc3_b
    
    return logits


# In[22]:


### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet_Updated(x): 
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x24.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 24), mean = 0, stddev = 0.1))
    conv1_b = tf.Variable(tf.zeros(24))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    
    # Pooling. Input = 28x28x24. Output = 14x14x24.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Activation 1.
    conv1 = tf.nn.relu(conv1)
    
    
    # Layer 2: Convolutional. Input = 14x14x24. Output = 10x10x64.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 24, 64), mean = 0, stddev = 0.1))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Pooling. Input = 10x10x64. Output = 5x5x64.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # Activation 2.
    conv2 = tf.nn.relu(conv2)
    #dropout
    conv2_dropout = tf.nn.dropout(conv2,keep_prob=0.5)

    
    # Flatten. Input = 5x5x64. Output = 1600.
    flattened   = flatten(conv2_dropout)
    
    #Matrix multiplication
    #input: 1x1600
    #weight: 1600x480 
    #Matrix multiplication(dot product rule)
    #output = 1x1600 * 1600x480  => 1x480
    
     # Layer 3: Fully Connected. Input = 1600. Output = 480.
    fullyc1_W = tf.Variable(tf.truncated_normal(shape=(1600, 480), mean = 0, stddev = 0.1))
    fullyc1_b = tf.Variable(tf.zeros(480))
    fullyc1   = tf.matmul(flattened, fullyc1_W) + fullyc1_b
    
    # Full connected layer activation 1.
    fullyc1    = tf.nn.relu(fullyc1)
    
    # Layer 4: Fully Connected. Input = 480. Output = 43.
    fullyc2_W  = tf.Variable(tf.truncated_normal(shape=(480, 43), mean = 0, stddev = 0.1))
    fullyc2_b  = tf.Variable(tf.zeros(43))
    logits    = tf.matmul(fullyc1, fullyc2_W) + fullyc2_b
    
    # Full connected layer activation 2.
    #fullyc2    = tf.nn.relu(fullyc2)
    
    # Layer 5: Fully Connected. Input = 84. Output = 43.
    #fullyc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = 0, stddev = 0.1))
    #fullyc3_b  = tf.Variable(tf.zeros(43))
    #logits = tf.matmul(fullyc2, fullyc3_W) + fullyc3_b
    
    return logits


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[23]:


#Hyper parameters
learning_rate = 1.0e-4
#epochs = 50
epochs = 100
batch_size = 64

#learning_rate = .001
#epochs = 40
#batch_size = 64


# In[24]:


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

logits = LeNet_Updated(x)
#logits = LeNet(x)
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)


# In[25]:


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver1 = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_loss/num_examples, total_accuracy/num_examples


# In[27]:


losses = {'train':[], 'validation':[]}

accuracies = {'train':[], 'validation':[]}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for counter, image in enumerate(X_train):
            image = augment_image(image,brightness=0.7, angle=10, translation=5, shear=2)
        X_train_normalized = (X_train-X_train.mean())/X_train.std()
         
        
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            #batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            
            batch_x, batch_y = X_train_normalized[offset:end], y_train[offset:end]
            
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
    
        
        #valid_loss, valid_accuracy = evaluate(X_valid, y_valid)
        
        train_loss, train_accuracy = evaluate(X_train_normalized, y_train)
        
        valid_loss, valid_accuracy = evaluate(X_valid_normalized, y_valid)
        
        print("Epoch {}, Train loss = {:.3f}, Train Accuracy = {:.3f}, Validation loss = {:.3f}, Validation Accuracy = {:.3f}".format(i+1,train_loss,train_accuracy,valid_loss,valid_accuracy))
        print()
        
        losses['train'].append(train_loss)
        losses['validation'].append(valid_loss)
        
        accuracies['train'].append(train_accuracy)
        accuracies['validation'].append(valid_accuracy)
        
    saver1.save(sess, './trafficlight_classifier_model')
    print("Model saved")


# In[87]:


def plot_learning_curve(epochs,accuracies):
    train_accuracy = accuracies['train']
    validtn_accuracy = accuracies['validation']
    epochs_list = np.linspace(1,100,num=100)
    #print(epochs_list)
    plt.figure(figsize=(20, 10))
    plt.plot(epochs_list, train_accuracy, label='train')
    plt.plot(epochs_list, validtn_accuracy, label='validation')
    plt.title('Learning Curve')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.xticks(epochs_list)
    plt.legend(loc='center right')
    plt.show()
    
plot_learning_curve(epochs,accuracies)   


# In[88]:


def plot_losses(epochs,losses):
    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['validation'], label='Validation loss')
    plt.legend()
    plt.ylim(ymax=0.5)
    
plot_losses(epochs,losses)    


# In[28]:


with tf.Session() as sess:
    saver1.restore(sess, tf.train.latest_checkpoint('.'))
    
    X_test, y_test = shuffle(X_test, y_test)
    for counter, image in enumerate(X_test):
        image = augment_image(image,brightness=0.7, angle=10, translation=5, shear=2)
    X_test_normalized = (X_test-X_test.mean())/X_test.std()

    test_loss, test_accuracy = evaluate(X_test_normalized, y_test)
    print("Test Loss = {:.3f}, Test Accuracy = {:.3f}".format(test_loss, test_accuracy))


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[73]:


### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import glob
import matplotlib.image as mpimg

#New_Signs = np.array(glob.glob('trafficsigns_german/RandomImages/traffic_sign_*.jpg') + 
                 #glob.glob('trafficsigns_german/RandomImages/traffic_sign_*.png'))


New_Signs = np.array(glob.glob('trafficsigns_german/new_imgs/selected_signs/traffic_sign_*.jpg') + 
                 glob.glob('trafficsigns_german/new_imgs/selected_signs/traffic_sign_*.png'))


new_images = [plt.imread(path) for path in New_Signs]

imageslist = []

#cols =5
cols =3
num_images = len(new_images)
rows = num_images//cols
plt.figure(figsize=(cols*3,rows*2.5))

sign_names_list = np.asarray([13,3,11,4,9,10])


print('-' * 80)
print('New Images for Random Testing')
print('-' * 80)


for i, image in enumerate(new_images):
    resized_image = cv2.resize(cv2.imread(New_Signs[i]),(32,32))
    #imageslist.append(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    imageslist.append(resized_image)
    plt.subplot(rows,cols,i+1)
    plt.imshow(image)
    plt.text(0, 0, '{}: {}'.format(i, df.Name[sign_names_list[i]]), color='k',backgroundcolor='y', fontsize=8)        
    plt.text(0, image.shape[0], '{}'.format(image.shape), color='k',backgroundcolor='c', fontsize=8)   
    plt.xticks([])
    plt.yticks([])    
plt.show()


inputdata = np.asarray(imageslist)


# ### Predict the Sign Type for Each Image

# In[74]:


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
def resize_image(image): 
    return cv2.resize(image,(32,32))

def show_preprocessed_images(imagelist):
    #cols =5
    cols =3
    num_images = len(new_images)
    rows = num_images//cols
    plt.figure(figsize=(cols*3,rows*2.5))
   
    for i, image in enumerate(imagelist):
        #image = resize_image(image)
        image = augment_image(image,brightness=0.7, angle=10, translation=5, shear=2)
        image = (image-image.mean())/image.std()
        plt.subplot(rows,cols,i+1)
        plt.imshow(image)
        plt.text(0, 0, '{}: {}'.format(i, df.Name[sign_names_list[i]]), color='k',backgroundcolor='y', fontsize=8)        
        plt.text(0, image.shape[0], '{}'.format(image.shape), color='k',backgroundcolor='c', fontsize=8)   
        plt.xticks([])
        plt.yticks([])    
    plt.show()
    
    return imagelist
    


    
preprocessed_imaglist = show_preprocessed_images(inputdata)


# ### Analyze Performance

# In[78]:


### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
#inputdatanew = (inputdata-inputdata.mean())/inputdata.std()  
def eval_prediction(input_data, batch):
    steps_per_epoch = len(input_data) // batch + (len(input_data)%batch > 0)
    sess = tf.get_default_session()
    predictions = np.zeros((len(input_data), n_classes))
    
    for i, image in enumerate(input_data):
            #image = resize_image(image)
            image = augment_image(image,brightness=0.7, angle=10, translation=5, shear=2)
            #image = (image-image.mean())/image.std()
    
    input_data = (input_data-input_data.mean())/input_data.std()   
    
    for step in range(steps_per_epoch):      
        batch_x = input_data[step*batch:(step+1)*batch]
        batch_y = np.zeros((len(batch_x), n_classes))
        prediction = sess.run(tf.nn.softmax(logits), feed_dict={x: batch_x})
        predictions[step*batch:(step+1)*batch] = prediction
    return predictions

pred_result = None

with tf.Session() as sess:
    saver1.restore(sess, tf.train.latest_checkpoint('.'))
    prediction = eval_prediction(inputdata, 64)
    pred_result = sess.run(tf.nn.top_k(tf.constant(prediction),k=5))
    values, indices = pred_result
    print("Output")
    for each in indices:
        print('{} => {}'.format(each[0], df.Name[each[0]]))


# In[79]:


with tf.Session() as sess:
    saver1.restore(sess, tf.train.latest_checkpoint('.'))

    accuracy = sess.run(accuracy_operation, feed_dict={
        x: inputdata,
        y: sign_names_list,
    })

    print('Accuracy: {:.6f}'.format(accuracy))


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[80]:


### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
for i, image in enumerate(new_images):
    #image = augment_image(image,brightness=0.7, angle=10, translation=5, shear=2)
    #image = (image-image.mean())/image.std()
    print("Values:")
    for j in values[i]:
        print('{:.6f}'.format(j))
    print("Indices:")
    for j in indices[i]:
        print('{}: {}'.format(j, df.Name[j]))
    plt.imshow(image,aspect="auto")
    plt.show()
    print("-----------------------------------------------------------------------------")


# In[89]:


from sklearn.preprocessing import LabelBinarizer


# In[281]:


def display_image_predictions(original_images,correct_ids,features, labels, predictions):
    n_classes = len(labels)
    label_names = labels
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(n_classes))
    #label_ids = label_binarizer.inverse_transform(np.array(labels))
    label_ids = label_binarizer.inverse_transform(np.vstack(labels))
      
    fig, axies = plt.subplots(nrows=6, ncols=2,figsize=(20,10))
    fig.tight_layout()
    fig.suptitle('Softmax Predictions', fontsize=40, y=1.1)

    n_predictions = 5
    margin = 0.05
    #margin = 0.0005
    ind = np.arange(n_predictions)
    width = (1. - 2. * margin) / n_predictions
    
    for image_i, (feature, label_id, pred_indicies, pred_values,correctid) in enumerate(zip(original_images, label_ids, predictions.indices, predictions.values,correct_ids)):
       
        pred_names = [label_names[pred_i] for pred_i in pred_indicies]
        
        correct_name = label_names[correctid]
           
        axies[image_i][0].imshow(feature)
        axies[image_i][0].set_title(correct_name)
        axies[image_i][0].set_axis_off()

        axies[image_i][1].barh(ind + margin, pred_values[::-1], width)
        axies[image_i][1].set_yticks(ind + margin)
        axies[image_i][1].set_yticklabels(pred_names[::-1])
        axies[image_i][1].set_xticks([0, 0.5, 1.0])


# In[282]:


traffic_sign_names_list = pd.read_csv('signnames.csv', index_col='ClassId')

signlist =[]

for sign in traffic_sign_names_list['SignName']:
    signlist.append(sign)

display_image_predictions(new_images,sign_names_list,inputdata, signlist, pred_result) 


# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

# In[ ]:





# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[ ]:


### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

