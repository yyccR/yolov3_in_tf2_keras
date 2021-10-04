import tensorflow as tf


class DarkNet:
    def __init__(self):
        pass

    def _darknet_conv(self, x, filters, size, strides=1, batch_norm=True):
        if strides == 1:
            padding = 'same'
        else:
            x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
            padding = 'valid'
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=size,
                                   strides=strides,
                                   padding=padding,
                                   use_bias=not batch_norm,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
        if batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        return x

    def _darknet_residual(self, x, filters):
        prev = x
        x = self._darknet_conv(x, filters // 2, 1)
        x = self._darknet_conv(x, filters, 3)
        x = tf.keras.layers.Add()([prev, x])
        return x

    def _darknet_block(self, x, filters, blocks):
        x = self._darknet_conv(x, filters, 3, strides=2)
        for _ in range(blocks):
            x = self._darknet_residual(x, filters)
        return x

    def build_darknet(self, x, name=None):
        # x = inputs = tf.keras.layers.Input([None, None, 3])
        x = self._darknet_conv(x, 32, 3)
        # 1/2
        x = self._darknet_block(x, 64, 1)
        # 1/4
        x = self._darknet_block(x, 128, 2)
        # 1/8
        x = x1 = self._darknet_block(x, 256, 8)
        # 1/16
        x = x2 = self._darknet_block(x, 512, 8)
        # 1/32
        x3 = self._darknet_block(x, 1024, 4)
        # return tf.keras.Model(inputs, (x_36, x_61, x), name=name)
        return x1, x2, x3

    def build_darknet_tiny(self, x, name=None):
        # x = inputs = tf.keras.layers.Input([None, None, 3])
        x = self._darknet_conv(x, 16, 3)
        x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
        x = self._darknet_conv(x, 32, 3)
        x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
        x = self._darknet_conv(x, 64, 3)
        x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
        x = self._darknet_conv(x, 128, 3)
        x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
        x = x_8 = self._darknet_conv(x, 256, 3)  # skip connection
        x = tf.keras.layers.MaxPool2D(2, 2, 'same')(x)
        x = self._darknet_conv(x, 512, 3)
        x = tf.keras.layers.MaxPool2D(2, 1, 'same')(x)
        x = self._darknet_conv(x, 1024, 3)
        # return tf.keras.Model(inputs, (x_8, x), name=name)
        return x_8, x

if __name__ == '__main__':
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    darknet = DarkNet()
    darknet_model = darknet.build_darknet('darknet')
    # darknet_tiny_model = darknet.build_darknet_tiny('darknet_tiny')
    # darknet_model.summary(line_length=200)
    # darknet_tiny_model.summary(line_length=100)
    # tf.keras.utils.plot_model(resnet_model)
    # from tensorflow.python.ops import summary_ops_v2
    # from tensorflow.python.keras.backend import get_graph
    # tb_writer = tf.summary.create_file_writer('./logs')
    # with tb_writer.as_default():
    #     if not darknet_model.run_eagerly:
    #         summary_ops_v2.graph(get_graph(), step=0)