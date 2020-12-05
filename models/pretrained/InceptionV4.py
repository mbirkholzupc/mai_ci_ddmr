import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers import Input, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model


class InceptionV4:
    _channel_axis = -1

    def get_model(self):
        inputs = Input((299, 299, 3))
        x = self._base_model(inputs)
        model = Model(inputs, x, name='inception_v4')
        model.load_weights(os.path.join(os.path.dirname(os.path.realpath(__file__)),
            'weights', 'inception-v4_weights_tf_dim_ordering_tf_kernels_notop.h5'
        ),
            by_name=True)
        return model

    def _base_model(self, input):
        # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
        net = self._conv2d_bn(input, 32, 3, 3, strides=(2, 2), padding='valid')
        net = self._conv2d_bn(net, 32, 3, 3, padding='valid')
        net = self._conv2d_bn(net, 64, 3, 3)

        branch_0 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)

        branch_1 = self._conv2d_bn(net, 96, 3, 3, strides=(2, 2), padding='valid')

        net = concatenate([branch_0, branch_1], axis=self._channel_axis)

        branch_0 = self._conv2d_bn(net, 64, 1, 1)
        branch_0 = self._conv2d_bn(branch_0, 96, 3, 3, padding='valid')

        branch_1 = self._conv2d_bn(net, 64, 1, 1)
        branch_1 = self._conv2d_bn(branch_1, 64, 1, 7)
        branch_1 = self._conv2d_bn(branch_1, 64, 7, 1)
        branch_1 = self._conv2d_bn(branch_1, 96, 3, 3, padding='valid')

        net = concatenate([branch_0, branch_1], axis=self._channel_axis)

        branch_0 = self._conv2d_bn(net, 192, 3, 3, strides=(2, 2), padding='valid')
        branch_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)

        net = concatenate([branch_0, branch_1], axis=self._channel_axis)

        # 35 x 35 x 384
        # 4 x Inception-A blocks
        for idx in range(4):
            net = self._block_inception_a(net)

        # 35 x 35 x 384
        # Reduction-A block
        net = self._block_reduction_a(net)

        # 17 x 17 x 1024
        # 7 x Inception-B blocks
        for idx in range(7):
            net = self._block_inception_b(net)

        # 17 x 17 x 1024
        # Reduction-B block
        net = self._block_reduction_b(net)

        # 8 x 8 x 1536
        # 3 x Inception-C blocks
        for idx in range(3):
            net = self._block_inception_c(net)

        return net

    def _conv2d_bn(self, x, nb_filter, num_row, num_col,
                   padding='same', strides=(1, 1), use_bias=False):
        """
        Utility function to apply conv + BN.
        (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
        """
        x = Convolution2D(nb_filter, (num_row, num_col),
                          strides=strides,
                          padding=padding,
                          use_bias=use_bias)(x)
        x = BatchNormalization(axis=self._channel_axis, scale=False)(x)
        x = Activation('relu')(x)
        return x

    def _block_inception_a(self, input):
        branch_0 = self._conv2d_bn(input, 96, 1, 1)

        branch_1 = self._conv2d_bn(input, 64, 1, 1)
        branch_1 = self._conv2d_bn(branch_1, 96, 3, 3)

        branch_2 = self._conv2d_bn(input, 64, 1, 1)
        branch_2 = self._conv2d_bn(branch_2, 96, 3, 3)
        branch_2 = self._conv2d_bn(branch_2, 96, 3, 3)

        branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
        branch_3 = self._conv2d_bn(branch_3, 96, 1, 1)

        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=self._channel_axis)
        return x

    def _block_reduction_a(self, input):
        branch_0 = self._conv2d_bn(input, 384, 3, 3, strides=(2, 2), padding='valid')

        branch_1 = self._conv2d_bn(input, 192, 1, 1)
        branch_1 = self._conv2d_bn(branch_1, 224, 3, 3)
        branch_1 = self._conv2d_bn(branch_1, 256, 3, 3, strides=(2, 2), padding='valid')

        branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

        x = concatenate([branch_0, branch_1, branch_2], axis=self._channel_axis)
        return x

    def _block_inception_b(self, input):
        branch_0 = self._conv2d_bn(input, 384, 1, 1)

        branch_1 = self._conv2d_bn(input, 192, 1, 1)
        branch_1 = self._conv2d_bn(branch_1, 224, 1, 7)
        branch_1 = self._conv2d_bn(branch_1, 256, 7, 1)

        branch_2 = self._conv2d_bn(input, 192, 1, 1)
        branch_2 = self._conv2d_bn(branch_2, 192, 7, 1)
        branch_2 = self._conv2d_bn(branch_2, 224, 1, 7)
        branch_2 = self._conv2d_bn(branch_2, 224, 7, 1)
        branch_2 = self._conv2d_bn(branch_2, 256, 1, 7)

        branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
        branch_3 = self._conv2d_bn(branch_3, 128, 1, 1)

        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=self._channel_axis)
        return x

    def _block_reduction_b(self, input):
        branch_0 = self._conv2d_bn(input, 192, 1, 1)
        branch_0 = self._conv2d_bn(branch_0, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_1 = self._conv2d_bn(input, 256, 1, 1)
        branch_1 = self._conv2d_bn(branch_1, 256, 1, 7)
        branch_1 = self._conv2d_bn(branch_1, 320, 7, 1)
        branch_1 = self._conv2d_bn(branch_1, 320, 3, 3, strides=(2, 2), padding='valid')

        branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

        x = concatenate([branch_0, branch_1, branch_2], axis=self._channel_axis)
        return x

    def _block_inception_c(self, input):
        branch_0 = self._conv2d_bn(input, 256, 1, 1)

        branch_1 = self._conv2d_bn(input, 384, 1, 1)
        branch_10 = self._conv2d_bn(branch_1, 256, 1, 3)
        branch_11 = self._conv2d_bn(branch_1, 256, 3, 1)
        branch_1 = concatenate([branch_10, branch_11], axis=self._channel_axis)

        branch_2 = self._conv2d_bn(input, 384, 1, 1)
        branch_2 = self._conv2d_bn(branch_2, 448, 3, 1)
        branch_2 = self._conv2d_bn(branch_2, 512, 1, 3)
        branch_20 = self._conv2d_bn(branch_2, 256, 1, 3)
        branch_21 = self._conv2d_bn(branch_2, 256, 3, 1)
        branch_2 = concatenate([branch_20, branch_21], axis=self._channel_axis)

        branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
        branch_3 = self._conv2d_bn(branch_3, 256, 1, 1)

        x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=self._channel_axis)
        return x
