import tensorflow as tf
import numpy as np
import math

class layersObj:

    #define functions like conv, deconv, upsample etc inside this class

    def __init__(self):
        #print('init')
        self.batch_size=20
        #self.batch_size=cfg.batch_size

    def conv2d_layer(self,ip_layer,    # The previous/input layer.
                 name,                 # Name of the conv layer
                 bias_init=0,          # Constant bias value for initialization
                 kernel_size=(3,3),    # Width and height of each filter.
                 strides=(1,1),        # stride value of the filter
                 num_filters=32,       # Number of output filters.
                 padding='SAME',       # Padding - SAME for zero padding - i/p and o/p of conv. have same dimensions
                 use_bias=True,        #to use bias or not
                 use_relu=True,        # Use relu as activation function
                 use_batch_norm=False,# use batch norm on layer before passing to activation function
                 use_conv_stride=False,  # Use 2x2 max-pooling - obtained by convolution with stride 2.
                 training_phase=True,   # Training Phase
                 scope_name=None,      # scope name for batch norm
                 acti_type='xavier',   # weight and bias variable initializer type
                 dilated_conv=False,   # dilated convolution enbale/disable
                 dilation_factor=1):   # dilation factor
        '''
        Standard 2D convolutional layer
        '''
        # Num. channels in prev. layer.
        prev_layer_no_filters = ip_layer.get_shape().as_list()[-1]

        weight_shape = [kernel_size[0], kernel_size[1], prev_layer_no_filters, num_filters]
        bias_shape = [num_filters]

        strides_augm = [1, strides[0], strides[1], 1]

        if(scope_name==None):
            scope_name=str(name)+'_bn'

        with tf.variable_scope(name):

            weights = self.get_weight_variable(weight_shape, name='W',acti_type=acti_type)
            if(use_bias==True):
                biases = self.get_bias_variable(bias_shape, name='b', init_bias_val=bias_init)

            if(dilated_conv==True):
                op_layer = tf.nn.atrous_conv2d(ip_layer, filters=weights, rate=dilation_factor, padding=padding, name=name)
            else:
                if(use_conv_stride==False):
                    op_layer = tf.nn.conv2d(ip_layer, filter=weights, strides=strides_augm, padding=padding)
                else:
                    op_layer = tf.nn.conv2d(input=ip_layer, filter=weights, strides=[1, 2, 2, 1], padding=padding)

            #Add bias
            if(use_bias==True):
                op_layer = tf.nn.bias_add(op_layer, biases)

            if(use_batch_norm==True):
                op_layer = self.batch_norm_layer(ip_layer=op_layer,name=scope_name,training=training_phase)
            if(use_relu==True):
                op_layer = tf.nn.relu(op_layer)

            # Add Tensorboard summaries
            #_add_summaries(op_layer, weights, biases)

        #return op_layer,weights,biases
        return op_layer

    def deconv2d_layer(self,ip_layer,   # The previous layer.
                 name,                  # Name of the conv layer
                 bias_init=0,           # Constant bias value for initialization
                 kernel_size=(3,3),     # Width and height of each filter.
                 strides=(2,2),         # stride value of the filter
                 num_filters=32,        # Number of filters.
                 padding='SAME',        # Padding - SAME for zero padding - i/p and o/p of conv. have same dimensions
                 output_shape=None,     # output shape of deconv. layer
                 use_bias=True,         # to use bias or not
                 use_relu=True,         # Use relu as activation function
                 use_batch_norm=False,  # use batch norm on layer before passing to activation function
                 training_phase=True,   # Training Phase
                 scope_name=None,       #scope name for batch norm
                 acti_type='xavier',dim_list=None):   #scope name for batch norm

        '''
        Standard 2D transpose (also known as deconvolution) layer. Default behaviour upsamples the input by a factor of 2.
        '''
        # Shape of prev. layer (aka. input layer)
        if(dim_list==None):
            prev_layer_shape = ip_layer.get_shape().as_list()
        else:
            prev_layer_shape = dim_list
        batch_size_val=tf.shape(ip_layer)[0]

        if output_shape is None:
            output_shape = tf.stack([tf.shape(ip_layer)[0], tf.shape(ip_layer)[1] * strides[0], tf.shape(ip_layer)[2] * strides[1], num_filters])

        # Num. channels in prev. layer.
        prev_layer_no_filters = prev_layer_shape[3]

        weight_shape = [kernel_size[0], kernel_size[1], num_filters, prev_layer_no_filters]
        bias_shape = [num_filters]
        strides_augm = [1, strides[0], strides[1], 1]

        if(scope_name==None):
            scope_name=str(name)+'_bn'

        with tf.variable_scope(name):

            weights = self.get_weight_variable(weight_shape, name='W',acti_type=acti_type)
            if(use_bias==True):
                biases = self.get_bias_variable(bias_shape, name='b', init_bias_val=bias_init)

            op_layer = tf.nn.conv2d_transpose(ip_layer,
                                        filter=weights,
                                        output_shape=output_shape,
                                        strides=strides_augm,
                                        padding=padding)

            #Add bias
            if(use_bias==True):
                op_layer = tf.nn.bias_add(op_layer, biases)

            if(use_batch_norm==True):
                op_layer = self.batch_norm_layer(ip_layer=op_layer,name=scope_name,training=training_phase)
            if(use_relu==True):
                op_layer = tf.nn.relu(op_layer)

            # Add Tensorboard summaries
            #_add_summaries(op_layer, weights, biases)

        #return op_layer,weights,biases
        return op_layer

    def lrelu(self, x, leak=0.2, name='lrelu'):
        # Leaky Relu layer
        return tf.maximum(x, leak*x)

    def upsample_layer(self,ip_layer, method=0, scale_factor=2, dim_list=None):
        '''
        2D upsampling layer with default image scale factor of 2.
        ip_layer : input feature map layer
        method = 0 --> Bilinear Interpolation
                 1 --> Nearest Neighbour
                 2 --> Bicubic Interpolation
        scale_factor : factor by which we want to upsample current resolution
        '''
        if(dim_list!=None):
            prev_height = dim_list[1]
            prev_width = dim_list[2]
        else:
            prev_height = ip_layer.get_shape().as_list()[1]
            prev_width = ip_layer.get_shape().as_list()[2]

        new_height = int(round(prev_height * scale_factor))
        new_width = int(round(prev_width * scale_factor))

        op = tf.image.resize_images(images=ip_layer,size=[new_height,new_width],method=method)

        return op

    def max_pool_layer2d(self,ip_layer, kernel_size=(2, 2), strides=(2, 2), padding="SAME", name=None):
        '''
        2D max pooling layer with standard 2x2 pooling with stride 2 as default
        '''

        kernel_size_aug = [1, kernel_size[0], kernel_size[1], 1]
        strides_aug = [1, strides[0], strides[1], 1]

        op = tf.nn.max_pool(ip_layer, ksize=kernel_size_aug, strides=strides_aug, padding=padding, name=name)

        return op

    def batch_norm_layer(self,ip_layer, name, training, moving_average_decay=0.99, epsilon=1e-3):
        '''
        Batch normalisation layer (Adapted from https://github.com/tensorflow/tensorflow/issues/1122)
        input params:
            ip_layer: Input layer (should be before activation)
            name:     A name for the computational graph
            training: A tf.bool specifying if the layer is executed at training or testing time
        returns:
            normalized: Batch normalised activation
        '''

        with tf.variable_scope(name):

            n_out = ip_layer.get_shape().as_list()[-1]
            tensor_dim = len(ip_layer.get_shape().as_list())

            if tensor_dim == 2:
                # must be a dense layer
                moments_over_axes = [0]
            elif tensor_dim == 4:
                # must be a 2D conv layer
                moments_over_axes = [0, 1, 2]
            elif tensor_dim == 5:
                # must be a 3D conv layer
                moments_over_axes = [0, 1, 2, 3]
            else:
                # is not likely to be something reasonable
                raise ValueError('Tensor dim %d is not supported by this batch_norm layer' % tensor_dim)

            init_beta = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
            init_gamma = tf.constant(1.0, shape=[n_out], dtype=tf.float32)
            beta = tf.get_variable(name='beta', dtype=tf.float32, initializer=init_beta, regularizer=None,
                                   trainable=True)
            gamma = tf.get_variable(name='gamma', dtype=tf.float32, initializer=init_gamma, regularizer=None,
                                    trainable=True)

            batch_mean, batch_var = tf.nn.moments(ip_layer, moments_over_axes, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=moving_average_decay)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(training, mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normalised = tf.nn.batch_normalization(ip_layer, mean, var, beta, gamma, epsilon)

        return normalised

    ### VARIABLE INITIALISERS ####################################################################################

    def get_weight_variable(self,shape,name=None,acti_type='xavier',fs=3):
        """
        Initializes the weights/convolutional kernels based on Xavier's method. Dimensions of filters are determined as per the input shape.
        Xavier's method initializes values of filters randomly around zero with the standard deviation computed as per Xavier's method.
        Args:
            shape : provides the shape of convolutional filter.
                    shape[0], shape[1] denotes the dimensions (height and width) of filters.
                    shape[2] denotes the depth of each filter. This also denotes the depth of each feature.
                    shape[3] denotes the number of filters which will determine the number of features that have to be computed.
        Returns:
            Weights/convolutional filters initialized as per Xavier's method. The dimensions of the filters are set as per input shape variable.
        """
        nInputUnits=shape[0]*shape[1]*shape[2]
        stddev_val = 1. / math.sqrt( nInputUnits/2 )
        #http://cs231n.github.io/neural-networks-2/#init
        return tf.Variable(tf.random_normal(shape, stddev=stddev_val, seed=1),name=name)

    def get_bias_variable(self,shape, name=None, init_bias_val=0.0):
        """
        Initializes the biases as per input bias_val. The initial value is equal to zero + bias_val.
        Number of such bias values required are determined by the number of filters which is input as length variable.
        Args:
            shape : provides us the number of filters.
            init_bias_val : provides the base bias_val to initialize all bias values with.
        Returns:
            biases initilialized as per the bias_val.
        """
        return tf.Variable(tf.zeros(shape=shape)+init_bias_val,name=name)

