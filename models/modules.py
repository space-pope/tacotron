import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMBlockCell
from util.ops import shape_list


def prenet(inputs, is_training, layer_sizes=[256, 128], scope=None):
  x = inputs
  drop_rate = 0.5 if is_training else 0.0
  with tf.variable_scope(scope or 'prenet'):
    for i, size in enumerate(layer_sizes):
      dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))
      x = tf.layers.dropout(dense, rate=drop_rate, name='dropout_%d' % (i+1))
  return x

def reference_encoder(inputs, filters, kernel_size, strides, encoder_cell, is_training, scope='ref_encoder'):
  with tf.variable_scope(scope):
    ref_outputs = tf.expand_dims(inputs,axis=-1)
    # CNN stack
    for i, channel in enumerate(filters):
      ref_outputs = conv2d(ref_outputs, channel, kernel_size, strides, tf.nn.relu, is_training, 'conv2d_%d' % i)

    shapes = shape_list(ref_outputs)
    ref_outputs = tf.reshape(
      ref_outputs,
      shapes[:-2] + [shapes[2] * shapes[3]])
    # RNN
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
      encoder_cell,
      ref_outputs,
      dtype=tf.float32)

    reference_state = tf.layers.dense(encoder_outputs[:,-1,:], 128, activation=tf.nn.tanh) # [N, 128]
    return reference_state


def encoder_cbhg(inputs, input_lengths, is_training):
  return cbhg(
    inputs,
    input_lengths,
    is_training,
    scope='encoder_cbhg',
    K=16,
    projections=[128, 128])


def post_cbhg(inputs, input_dim, is_training):
  return cbhg(
    inputs,
    None,
    is_training,
    scope='post_cbhg',
    K=8,
    projections=[256, input_dim])


def cbhg(inputs, input_lengths, is_training, scope, K, projections):
  with tf.variable_scope(scope):
    ref_outputs = tf.expand_dims(inputs,axis=-1)
    # CNN stack
    for i, channel in enumerate(filters):
      ref_outputs = conv2d(ref_outputs, channel, kernel_size, strides, tf.nn.relu, is_training, 'conv2d_%d' % i)

    shapes = shape_list(ref_outputs)
    ref_outputs = tf.reshape(
      ref_outputs,
      shapes[:-2] + [shapes[2] * shapes[3]])
    # RNN
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
      encoder_cell,
      ref_outputs,
      dtype=tf.float32)

    reference_state = tf.layers.dense(encoder_outputs[:,-1,:], 128, activation=tf.nn.tanh) # [N, 128]
    return reference_state


def highwaynet(inputs, scope):
  with tf.variable_scope(scope):
    H = tf.layers.dense(
      inputs,
      units=128,
      activation=tf.nn.relu,
      name='H')
    T = tf.layers.dense(
      inputs,
      units=128,
      activation=tf.nn.sigmoid,
      name='T',
      bias_initializer=tf.constant_initializer(-1.0))
    return H * T + inputs * (1.0 - T)


def conv_and_lstm(inputs, input_lengths, conv_layers, conv_width, conv_channels, lstm_units,
                  is_training, scope):
  # Convolutional layers
  with tf.variable_scope(scope):
    x = inputs
    for i in range(conv_layers):
      activation = tf.nn.relu if i < conv_layers - 1 else None
      x = conv1d(x, conv_width, conv_channels, activation, is_training, 'conv_%d' % i)

    # 2-layer bidirectional LSTM:
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
      LSTMBlockCell(lstm_units),
      LSTMBlockCell(lstm_units),
      x,
      sequence_length=input_lengths,
      dtype=tf.float32,
      scope='encoder_lstm')

    # Concatentate forward and backwards:
    return tf.concat(outputs, axis=2)


def postnet(inputs, layers, conv_width, channels, is_training):
  x = inputs
  with tf.variable_scope('decoder_postnet'):
    for i in range(layers):
      activation = tf.nn.tanh if i < layers - 1 else None
      x = conv1d(x, conv_width, channels, activation, is_training, 'postnet_conv_%d' % i)
  return tf.layers.dense(x, inputs.shape[2])   # Project to input shape


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
  with tf.variable_scope(scope):
    conv1d_output = tf.layers.conv1d(
      inputs,
      filters=channels,
      kernel_size=kernel_size,
      activation=activation,
      padding='same')
    return tf.layers.batch_normalization(conv1d_output, training=is_training)

def conv2d(inputs, filters, kernel_size, strides, activation, is_training, scope):
  with tf.variable_scope(scope):
    conv2d_output = tf.layers.conv2d(
      inputs,
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding='same')
    conv2d_output = tf.layers.batch_normalization(conv2d_output, training=is_training)
    if activation is not None:
      conv2d_output = activation(conv2d_output)
    return conv2d_output
