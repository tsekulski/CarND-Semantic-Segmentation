import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from time import time

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    #load VGG16 model graph from file
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    #grab the graph in a variable
    graph = tf.get_default_graph()
    
    #extract outputs (tensors) from selected VGG16 layers as specified above
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    
    # 1x1 convolution instead of fully connected layer - in order to preserve spatial information
    # I apply 1x1 convolution to each vgg layer in order to align the output dimensions (num_classes)
    # for the skip connections later on,
    # and also to make an intermediate "prediction" of the two classes at each of the vgg layers.
    
    vgg_layer_7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                bias_initializer=tf.random_normal_initializer(stddev=0.01), 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    vgg_layer_4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                bias_initializer=tf.random_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    vgg_layer_3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                bias_initializer=tf.random_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    #upsample 2x
    vgg_layer_7_upsampled = tf.layers.conv2d_transpose(vgg_layer_7_conv_1x1, num_classes, 4, 2, padding='same',
                                activation = tf.nn.relu,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                bias_initializer=tf.random_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    #add skip connection - by combining output of two layers
    vgg_layer_4_skip = tf.add(vgg_layer_7_upsampled, vgg_layer_4_conv_1x1)
    
    #upsample 2x
    vgg_layer_4_upsampled = tf.layers.conv2d_transpose(vgg_layer_4_skip, num_classes, 4, 2, padding='same',
                                activation = tf.nn.relu,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                bias_initializer=tf.random_normal_initializer(stddev=0.01), 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    #add skip connection
    vgg_layer_3_skip = tf.add(vgg_layer_4_upsampled, vgg_layer_3_conv_1x1)
    
    #upsample 8x
    nn_last_layer = tf.layers.conv2d_transpose(vgg_layer_3_skip, num_classes, 16, 8, padding='same',
                                activation = tf.nn.relu,
                                kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                bias_initializer=tf.random_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)) 
    return nn_last_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # The final step is to define a loss.
    # That way, we can approach training a FCN just like we would approach training a normal classification CNN.
    # In the case of a FCN, the goal is to assign each pixel to the appropriate class.
    # We already happen to know a great loss function for this setup, cross entropy loss! 
    # Remember the output tensor is 4D so we have to reshape it to 2D.
    
    # Reshape nn_last_layer & labels
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    
    # Logits is now a 2D tensor where each row represents a pixel and each column a class.
    # From here we can just use standard cross entropy loss.
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    # Define optimizer and training operation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    
    # check training time
    t0 = time()
    
    print("Training...")
    print()
    
    for epoch in range(epochs):
        print("EPOCH {} ...".format(epoch+1))
        
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0001})
        
            print("Loss = {:.3f}".format(loss))
        
        # print training time
        print ("Time elapsed so far:", round(time()-t0, 3), "s")
        print()
    
    print ("Total training time:", round(time()-t0, 3), "s")
tests.test_train_nn(train_nn)


def run():
    print("Running the run function...")
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    print("Downloading pretrained VGG16 model...")
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        print("Loading VGG16 graph...")
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        print("Creating FCN layers...")
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        
        correct_label = tf.placeholder(tf.int32) #just a placeholder since correct_label will come from get_batches_fn function
        learning_rate = tf.placeholder(tf.float32) #just a placeholder since learning_rate is defined manually in the train_nn function
        
        print("Building TensorFLow loss and optimizer operations...")
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        epochs = 2
        batch_size = 5
        
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, 
                 correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()