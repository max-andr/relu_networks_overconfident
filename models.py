import tensorflow as tf
import numpy as np

from tensorflow.contrib.slim.nets import resnet_v2 as slim_resnet
# from tensorflow.contrib.slim.nets.resnet_v2 import resnet_v2_block, resnet_v2, resnet_arg_scope
# import tensorflow.contrib.slim.nets as slim_resnet


class BasicNN:
    """
    An abstract class for a Neural Network model.
    """

    def __init__(self, hps):
        self.hps = hps
        self.activation = hps.activation
        self.f, self.grads_f_x, self.hidden1 = None, None, None
        self.weights_layer, self.biases_layer, self.bn_layer = 0, 0, 0

    def dropout(self, x):
        x = tf.layers.dropout(x, 0.5, training=self.flag_train)
        return x

    def batch_norm(self, x):
        self.bn_layer += 1
        x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=1e-5, center=True, scale=True,
                                          training=self.flag_train)
        # x = tf.contrib.layers.batch_norm(x, decay=0.99, epsilon=1e-5, center=True, scale=True,
        #                                  is_training=self.flag_train, updates_collections=None)
        return x

    def fc_layer(self, name, x, n_out, bn=False, last=False):
        """FullyConnected layer for final output."""
        with tf.variable_scope(name):
            if len(x.shape) == 4:
                n_in = int(x.shape[1]) * int(x.shape[2]) * int(x.shape[3])
                x = tf.reshape(x, [-1, n_in])
            else:
                n_in = int(x.shape[1])
            init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / n_in))
            w = tf.get_variable('weights', [n_in, n_out], initializer=init)
            b = tf.get_variable('biases', [n_out], initializer=tf.constant_initializer(0.0))

            x = tf.nn.xw_plus_b(x, w, b)
            x = self.batch_norm(x) if bn else x
            x = self.activation(x) if not last else x
        return x

    def get_loss(self, logits, labels):
        return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)) #sum instead of mean since later we average the loss over the full batch including noise.


class CNN(BasicNN):
    def __init__(self, hps):
        super().__init__(hps)

    @staticmethod
    def _stride_arr(stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def weight_variable(self, shape):
        """ Creates a weight variable of a given shape *for conv layer*
            First `hps.n_random_layers` will be initialized randomly but not trained.
        """
        self.weights_layer += 1  # just for counting purposes
        n_in = int(shape[0]) * int(shape[1]) * int(shape[2])

        # Mainstream init
        init = tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / n_in))
        weights = tf.get_variable('weights', shape, tf.float32, initializer=init, trainable=True)
        return weights

    def bias_variable(self, shape):
        """ Creates a bias variable of a given shape.
            First `hps.n_random_layers` will be initialized randomly but not trained.
        """
        init = tf.constant_initializer(0.0)
        return tf.get_variable('biases', shape, initializer=init)

    @staticmethod
    def max_pool(x, size, stride):
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')

    @staticmethod
    def avg_pool(x, size, stride):
        return tf.nn.avg_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')

    def _conv(self, name, x, filter_size, in_filters, out_filters, stride, biases=False):
        """Convolution."""
        with tf.variable_scope(name):
            kernel = self.weight_variable([filter_size, filter_size, in_filters, out_filters])
            x = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding='SAME')
            if biases:
                x += self.bias_variable([out_filters])
            return x

    @staticmethod
    def _global_avg_pool(x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

    def conv_layer(self, name, x, size, n_out, stride, bn=False, biases=True):
        with tf.variable_scope(name):
            n_in = x.shape[-1]
            x = self._conv(name, x, size, n_in, n_out, stride, biases=biases)
            x = self.batch_norm(x) if bn else x
            x = self.activation(x)
        return x


class ResNet(CNN):
    def __init__(self, hps):
        """ResNet constructor.
        ResNet model. Based on Ritchie Ng ResNet model: https://github.com/ritchieng/resnet-tensorflow
        Related papers:
        https://arxiv.org/pdf/1512.03385v1.pdf - main paper
        https://arxiv.org/pdf/1603.05027v2.pdf - Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1605.07146v1.pdf - wide residual networks

        Args:
          flag_train: tf.bool() which is True when we run the comp. graph for training, False for testing
        """
        super().__init__(hps)

        self.use_bottleneck = False
        if self.use_bottleneck:
            self.res_func = self._bottleneck_residual
        else:
            self.res_func = self._residual

    def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self.batch_norm(x)
                x = self.activation(x)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self.batch_norm(x)
                x = self.activation(x)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self.batch_norm(x)
            x = self.activation(x)
            x = self._conv('conv2', x, 3, out_filter, out_filter, 1)

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, self._stride_arr(stride), self._stride_arr(stride), 'VALID')
                orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0],
                                         [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.info('image after unit %s', x.get_shape())
        return x

    def _bottleneck_residual(self, x, in_filter, out_filter, stride,
                             activate_before_residual=False):
        """Bottleneck resisual unit with 3 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('common_bn_relu'):
                x = self.batch_norm(x)
                x = self.activation(x)
                orig_x = x
        else:
            with tf.variable_scope('residual_bn_relu'):
                orig_x = x
                x = self.batch_norm(x)
                x = self.activation(x)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 1, in_filter, out_filter / 4, stride)

        with tf.variable_scope('sub2'):
            x = self.batch_norm(x)
            x = self.activation(x)
            x = self._conv('conv2', x, 3, out_filter / 4, out_filter / 4, [1, 1, 1, 1])

        with tf.variable_scope('sub3'):
            x = self.batch_norm(x)
            x = self.activation(x)
            x = self._conv('conv3', x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
            x += orig_x

        tf.logging.info('image after unit %s', x.get_shape())
        return x


class ResNetSmall(ResNet):
    def __init__(self, hps):
        super().__init__(hps)
        self.n_filters = [16, 16, 32, 64]
        # self.n_filters = [2, 2, 4, 8]
        # self.n_filters = [64, 64, 128, 256]

    def get_logits(self, x, flag_train):
        self.flag_train = flag_train
        res_func = self.res_func

        strides = [1, 1, 2, 2]
        activate_before_residual = [True, False, False]
        n_resid_units = [0, 3, 3, 3]

        with tf.variable_scope('block_init'):
            x = self._conv('conv', x, 3, int(x.shape[-1]), self.n_filters[0], strides[0])

        for i in range(1, len(n_resid_units)):
            with tf.variable_scope('block_' + str(i) + '_0'):
                x = res_func(x, self.n_filters[i - 1], self.n_filters[i], strides[i], activate_before_residual[0])
            for j in range(1, n_resid_units[i]):
                with tf.variable_scope('block_' + str(i) + '_' + str(j)):
                    x = res_func(x, self.n_filters[i], self.n_filters[i], 1, False)

        with tf.variable_scope('unit_last'):
            x = self.batch_norm(x)
            x = self.activation(x)
            x = self._global_avg_pool(x)

        with tf.variable_scope('logit'):
            f = self.fc_layer('fc', x, self.hps.n_classes, last=True)

        return f


def he_init(shape):
    """ He init for conv or fc layers."""
    if len(shape) == 4:
        n_in, n_out = shape[0] * shape[1] * shape[2], shape[0] * shape[1] * shape[3]
    else:
        n_in, n_out = shape[0], shape[1]
    return tf.truncated_normal(shape, stddev=np.sqrt(2.0 / n_in))


class LeNet(CNN):
    def __init__(self, hps):
        super().__init__(hps)
        self.strides = [1, 1]
        self.n_filters = [32, 64]
        self.n_fc = [1024]

    def get_logits(self, x, flag_train):
        """
        Build the core model within the graph.
          x: Batches of images. [batch_size, image_size, image_size, 3]
        """
        self.flag_train = flag_train
        bn = False

        x = self.conv_layer('conv1', x, 5, self.n_filters[0], self.strides[0], bn=bn, biases=not bn)
        x = self.max_pool(x, 2, 2)
        x = self.conv_layer('conv2', x, 5, self.n_filters[1], self.strides[1], bn=bn, biases=not bn)
        x = self.max_pool(x, 2, 2)
        x = self.fc_layer('fc1', x, self.n_fc[0])
        x = self.fc_layer('fc2', x, self.hps.n_classes, last=True)

        return x


models_dict = {'lenet': LeNet,
               'resnet_small': ResNetSmall,  
               }
