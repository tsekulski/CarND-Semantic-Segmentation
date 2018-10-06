# Semantic Segmentation

_Technologies: Deep Learning, Fully Convolutional Networks, TensorFlow_

Labeled the pixels of a road in images using a Fully Convolutional Network (FCN).

_Part of the Self-Driving Car Engineer Nanodegree Program_

### Introduction
In this project, I labeled the pixels of a road in images using a Fully Convolutional Network (FCN). I modified/expanded the starter code as provided in Udacity's [CarND-Semantic-Segmentation](https://github.com/udacity/CarND-Semantic-Segmentation) repo.

### Solution approach
- The encoder of the network is based on VGG16 convolutional layers.
- The decoder uses transposed convolutions (deconvolutions) of VGG layers 3, 4 and 7, which upsample the output to the original image height and width.
- Two "skip connection" layers are used, connecting the output of vgg layers 3 and 4 (in the encoder) to the corresponding deconvolution layers in the decoder. These skip connections allow the network to use information from multiple resolutions.
- The fully connected layer is replaced with 1x1 convolutions. 1x1 convolutions are also used as a "preprocessing" step preparing the input into the "skip connection" layers. The 1x1 convolutions serve following purposes:\s\s
  * they preserve spatial information,
  * 1x1 convolutions applied to each "skip connection" vgg layer ensure alignment of the output dimensions (num_classes = 2),
  * as a side effect, 1x1 convolutions creat an intermediate "prediction" of the two classes (road, not road) at selected vgg layers - those "predictions" are then used as an input into the "skip connection" layers.
- After several runs I settled for a dropout rate of 0.5, initial learning rate of 0.0001, kernel_initializer = tf.random_normal_initializer(stddev=0.01) and kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3)). Adding kernel initializer and regularizer had a clear positive impact on the results.
- I also added a relu activation and bias_initializer = tf.random_normal_initializer(stddev=0.01), however these modifications did not have a visible impact on the quality of predictions.
- I trained the network over 12 epochs with batch size = 5.

### Results
- The model labels a great majority of "road" pixels, however it also quite often labels some additional non-road pixels as road.
- The results seem to be in line with rubric requirement: "A solution that is close to best would label at least 80% of the road and label no more than 20% of non-road pixels as road".
- It therefore seems that my model has a relatively high recall score (almost all true road pixels are labeled as "road"), while the precision is relatively lower (quite some non-road pixels are wrongly labeled as "road").

### Potential further improvements
- A key further improvement would be to try to improve the precision of the model, i.e. reduce the number of non-road pixels predicted as road.
- Additional exploration of hyperparameter space could be a good approach to improve the precision.
