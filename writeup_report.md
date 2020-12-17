# **Behavioral Cloning** 

## Writeup 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


<!-- [//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image" -->

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model architecture is largely based on the architecture recommended during the lecture materials, with the modification that batch normalization layers are added after each convolutional layer. The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains batch normalization layers in order to reduce overfitting. 

The model was trained and validated on the data set provided by Udacity to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

The training data supplied by Udacity was used to train the model. No additional data was collected for training the model.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use the convolution neural network model recommended in the lecture materials without any regularization effects. The MSE loss stopped decreasing significantly after about 4 epochs, so I trained for 5 epochs. After this, I found that the model performed well enough on the test track without any regularization effects. I then added batch normalization layers to help speed up training and apply a mild regularization effect to the network. After training the network with batch normalization layers, I tested it on track one. The network was able to steer the car successfully around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

|      Layer      |                 Description                 |
| :-------------: | :-----------------------------------------: |
|      Input      |              160x320x3 RGB image            |
|      Lambda     | Normalize pixel channel values to [-0.5,0.5]|
|   Cropping2D    | Remove pixels from top and bottom of image  |
| Convolution 5x5 |               outputs 31x158x24             |
|      RELU       |                                             |
|    BatchNorm    |                                             |
| Convolution 5x5 |               outputs 14x77x36              |
|      RELU       |                                             |
|    BatchNorm    |                                             |
| Convolution 5x5 |               outputs 5x37x48               |
|      RELU       |                                             |
|    BatchNorm    |                                             |
| Convolution 3x3 |               outputs 3x35x64               |
|      RELU       |                                             |
|    BatchNorm    |                                             |
| Convolution 3x3 |               outputs 1x33x64               |
|      RELU       |                                             |
|    BatchNorm    |                                             |
|     Flatten     |               outputs 2112                  |
| Fully connected |               outputs 100                   |
|      RELU       |                                             |
| Fully connected |               outputs 50                    |
|      RELU       |                                             |
| Fully connected |               outputs 10                    |
|      RELU       |                                             |
| Fully connected |               outputs 1                     |

#### 3. Creation of the Training Set & Training Process

I randomly shuffled the data set supplied by Udacity and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the fact that the training and validation losses stopped decreasing significantly after 5 epochs. I used an Adam optimizer so that manually tuning the learning rate wasn't necessary.
