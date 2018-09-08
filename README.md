# **Traffic Sign Recognition** 

## Writeup

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 39209
Number of testing examples  = 12630
Image data shape  = (32, 32, 3)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

A simple implementation was carried out, which generated random samples from the training set along with their corresponding label values. 

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because as the computational expense would be reduced. The neural network wouldn't gain much accuracy having color information hence it was a trade-off for a lighter copmutational load. 

I then decided to normalize the data between 0.1 and 0.9 to keep the computation numerically stable. This helped to scale down the disparity in the data within the defined scale. I obtained the normalized data by subtracting the values with the mean and then diving by the data's standard deviation.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

As established, Convolutional Neural Networks are best applicable for image classification problems, my network architecture is based off of Yann LeCun's LeNet architecture. The network has 2 convolutional layers and 3 fully connected layers. 

The primary difference as copmared to the LeNet architecture is the addition of dropout to the fully connected layers. This seemed to have taken care of the overfitting characteristic I was observing with my trained model. I was getting a training set accuracy of 97% but my test set accuracy was less than 90%. This pushed me to add dropout to my network architecture. Finally, I experimented a little and found that average pooling gave me a better validation score as compared to max pooling.
d at first.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

AdamOptimizer is used as the optimizer for the network based off of its reputation for being more efficient than Stochastic Gradient Descent (SGD). After plying around with the batch size, I settled with a size of 128. This was mainly selected based off of the computation time it took for training the network. I also adjusted the learning rate to 0.002 as I found the network to reach a good validation accuracy in relatively less number of Epochs of 12. Furthermore, increasing the number of Epochs did little to improve the accuracy of the model.

For the model hyperparameters, I stuck with a mean of 0 and standard deviation/sigma of 0.1. An important aspect of the model is trying to keep a mean of 0 and equal variance, so these hyperparameters attempt to follow this philosophy. I tried a few other standard deviations but found a smaller one did not really help, while a larger one vastly increased the training time necessary.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Initially I had implemented Yann LeCun's LeNet architecture without much modification. I found that the validation test accuracy was not meeting the expected levels. This is when I decided to use grayscale images as it would reduce the copmutational cost, and I could try increasing the number of epochs to attempt an improvement in the validation test accuracy. This did make a difference and made me realize the importance of pre-processing the training data. I then increased the learning rate as the validation accuracy of the network kept increasing and could not increase beyond a point as the number of epochs had run out. I still felt that the network validation accuracy could be improved further, which is when I tried to change the pooling methods. I found that the average pooling method gave me a better result than max pooling. Finally, I found that the network validation accuracy did not improve much beyond 12 epochs, which is what I finally set. 

The final CNN has 2 convolutional layers, 3 fully connected, a learning rate of .002 and a batch size of 128 with a couple of average pooling convolutional layers integrated as well. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

The five German traffic signs images from other online sources that I have added are: 1) a road work sign, 2) a yield sign, 3) a left turn ahead sign, 4) a 60 km/h sign and 5) a stop sign.

One difficulty may be due to the low resolution of the images. I think it will definitely be able to tell the 60 km/h sign is a speed sign (which are all circles), but it may struggle to figure out the exact speed. The left turn ahead sign should not be too difficult for it to do. The road work sign could potentially be a struggle, due to a few signs using the same overall triangular shape. If the classifier pays attention to the shapes in the center it should still do fine. I am hoping the stop sign and yield sign should do well - those are more unique than a lot of the signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The model is 60% accurate when it came to classifying the new images that I added. The model incorrectly identified the road work sign as a general caution sign. It also incorrectly classified the left turn ahead sign as a keep right sign.

The road work sign is completely mis-classified as none of the top five predictions are a road work sign. However, that being said, it did almost classify the left turn ahead sign as it was the second most probable classification according to the network. 

The reason for the drop in accuracy as compared to the test dataset is likely caused by the highly curated nature of the original dataset. In comparison, the images that were added by me had very poor resolution and were only quickly cropped and downsized from those I obtained from the internet.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model correctly classifies three out of the five signs. 

The first image which is a road work sign is entirely misclassified. None of the top five predicitons are road work signs, hence this label is something I would look into given more time.

The model correctly identifies the yield sign with a significant amount of certainty as depicted by the softmax probability percentages. The unique shape of the yield sign also aids the network in clearly distinguishing this sign.

The model incoreectly identifies a left turn ahead sign as a keep right sign. This could be explained by the close relation the two signs have in terms of the curves of the shapes on them. Having said this, the model was very close in identifying the sign correctly as it predicted it to be a left turn ahead sign as the second highest possibility that too with a very narrow margin. 

The model accurately classifies the final two signs which are a 60km/h speed limit sign and a stop sign with a fair amount of certainty as depicted by the softmax probability outputs.

On the whole, the model performed with a 60% accuracy on the data added by me. However, it came very close to predicting the left turn ahead sign correctly and only mis-classified it by a narrow margin. Hence, it is more like an 80% accuracy. Hence, I strongly believe that by augmenting some new data images for some of these labels and re-training the network would yield in a significant improvement in classification.
