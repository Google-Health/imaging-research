"""exAMD prediction network architecture.

Implementation of exAMD prediction network described in  "Predicting exudative
conversion in age related macular degeneration using deep learning".


Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sonnet as snt
import tensorflow as tf


class ExAmdNet(snt.AbstractModule):
  """Future exAMD prediction deep learning network.

  Takes as input either a grey-scale 3D OCT volume or a one-hot encoded
  segmentation map of a 3D OCT volume. See manuscript for architecture details.
  """

  def __init__(self,
               name='ex_amd_net'):
    """Initializes the model and parameters.

    Args:
      name: Variable name of module.
    """
    super(ExAmdNet, self).__init__(name=name)

    # Convolution parameters.
    self._filter_chs = 32
    self._bottleneck_chs = 32

  def _build(self, inputs, is_training=True):
    """Internal method to build the sonnet module.

    Args:
      inputs: tensor of batch input OCT or dense segmentation maps.
              OCT shape: [batch, 41, 450, 450, 1]
              Segmentation map shape: [batch, 41, 450, 450, 17]
      is_training: flag for model usage when training

    Returns:
      Output tensor of module. A tensor with size equal to
      number of classes.
    """
    net = inputs

    # First level.
    net = block(net, 'l1', self._filter_chs // 4,
                block_kernels=[(1, 3, 3), (1, 3, 3)])
    net = max_pool3d(net, pool_size=(1, 2, 2), strides=(1, 2, 2), name='l1_out')
    print('Shape after L1: %s' % net.shape.as_list())

    # Second level
    net = block(net, 'l2',
                channels_per_layer=self._filter_chs // 2)
    net = max_pool3d(net, pool_size=(1, 2, 2), strides=(1, 2, 2), name='l2_out')
    print('Shape after L2: %s' % net.shape.as_list())

    # Third level
    net = conv_1x1x1(net, self._bottleneck_chs * 4, 'l3_1x1x1')
    net = block(net, 'l3',
                channels_per_layer=self._filter_chs // 2)
    net = max_pool3d(net, pool_size=(2, 2, 2), strides=(2, 2, 2), name='l3_out')
    print('Shape after L3 level: %s' % net.shape.as_list())

    # Fourth level
    net = conv_1x1x1(net, self._bottleneck_chs * 4, 'l4_1x1x1')
    for i in range(2):
      net = block(net, 'l4_b%d' % (i+1),
                  channels_per_layer=self._filter_chs)
    net = max_pool3d(net, pool_size=(2, 2, 2), strides=(2, 2, 2), name='l4_out')
    print('Shape after L4 level: %s' % net.shape.as_list())

    # Fifth level
    net = conv_1x1x1(net, self._bottleneck_chs * 4, 'l5_1x1x1')
    for i in range(2):
      net = block(net, 'l5_b%d' % i,
                  channels_per_layer=self._filter_chs)
    net = max_pool3d(net, pool_size=(2, 2, 2), strides=(2, 2, 2), name='l5_out')
    print('Shape after L5 level: %s' % net.shape.as_list())

    # Sixth level
    net = conv_1x1x1(net, self._bottleneck_chs * 8, 'l6_1x1x1')
    for i in range(2):
      net = block(net, 'l6_b%d' % i,
                  channels_per_layer=self._filter_chs)
    print('Shape after L6 level: %s' % net.shape.as_list())

    # Output
    net = snt.Conv3D(output_channels=self._bottleneck_chs * 4,
                     kernel_shape=(1, 1, 1),
                     stride=1,
                     padding=snt.SAME,
                     name='final_1x1x1')(net)
    print('Output shape: %s' % net.shape.as_list())
    return net


def conv_3d(inputs,
            output_channels,
            kernel_shape,
            strides,
            name,
            activation=tf.nn.relu,
            use_bias=True):
  """Wraps sonnet 3D conv module with a nonlinear activation."""
  conv_out = snt.Conv3D(
      output_channels=output_channels,
      kernel_shape=kernel_shape,
      stride=strides,
      use_bias=use_bias,
      name=name)(
          inputs)
  return activation(conv_out)


def block(inputs,
          name_prefix,
          channels_per_layer,
          block_kernels=None,
          activation=tf.nn.relu,
          stride=1):
  """Consecutive convolution filters with skip connections."""
  if not block_kernels:
    # Full block length if not specified.
    block_kernels = [(1, 3, 3), (1, 3, 3), (3, 1, 1), (1, 3, 3), (1, 3, 3),
                     (3, 1, 1)]
  layer_stack = [inputs]
  for kernel in block_kernels:
    # Iterate through all kernels to construct a stack of intermediate
    # representations.
    layer_stack.append(
        conv_3d(
            inputs=layer_stack[-1],
            output_channels=channels_per_layer,
            kernel_shape=kernel,
            strides=stride,
            activation=activation,
            name='{}_{}'.format(name_prefix,
                                'x'.join([str(x) for x in kernel]))))
  # Concatenate all representations in the layer output as final output.
  output = tf.concat(layer_stack, axis=-1)
  return output


def max_pool3d(inputs, pool_size, strides, name):
  return tf.keras.layers.MaxPool3D(
      pool_size=pool_size, strides=strides, name=name)(
          inputs)


def conv_1x1x1(inputs, channels, name):
  return snt.Conv3D(output_channels=channels,
                    kernel_shape=(1, 1, 1),
                    stride=1,
                    padding=snt.SAME,
                    name=name)(inputs)

