import tensorflow as tf
import os

import constants as c

##
# Data
##

# read_and_decode() and inputs() taken from https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/how_tos/reading_data/fully_connected_reader.py
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # TODO (David): Fill in correct features from TFRecords
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'angle': tf.FixedLenFeature([], tf.int64),
        })

    # Convert image from a scalar string tensor to a uint8 tensor
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([c.FRAME_HEIGHT, c.FRAME_WIDTH])

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    angle = tf.cast(features['angle'], tf.int32)

    # Normalize from [0, 255] -> [-1, 1] floats.
    image = tf.cast(image, tf.float32) * (2. / 255) - 1

    # TODO (Mike): Data augmentation

    return image, angle

def get_inputs(train, batch_size, num_epochs=None):
    """Reads input data num_epochs times.

    Args:
      train: Selects between the training (True) and validation (False) data.
      batch_size: Number of examples per returned batch.
      num_epochs: Number of times to read the input data, or 0/None to
         train forever.

    Returns:
      A tuple (images, labels), where:
      * images is a float tensor with shape [batch_size, c.FRAME_HEIGHT, c.FRAME_WIDTH]
        in the range [-1, 1].
      * labels is an int32 tensor with shape [batch_size] with the true steering angle.
      Note that a tf.train.QueueRunner is added to the graph, which
      must be run using e.g. tf.train.start_queue_runners().
    """
    files = os.listdir(c.TRAIN_DIR) if train else os.listdir(c.VALIDATION_DIR)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(files,
                                                        num_epochs=num_epochs,
                                                        capacity=len(files))

        # Even when reading in multiple threads, share the filename queue.
        image, label = read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        # TODO (Matt): Make sure this capacity param is okay.
        images, sparse_labels = tf.train.shuffle_batch([image, label],
                                                       batch_size=batch_size,
                                                       num_threads=2,
                                                       capacity=1000 + 3 * batch_size,
                                                       min_after_dequeue=1000)

        return images, sparse_labels

##
# Loss
##

def MSE_loss(inputs, targets):
    """
    @return: The mean squared error between inputs and targets.
    """
    return tf.reduce_mean(tf.square(inputs - targets))

##
# Misc
##

# from http://stackoverflow.com/questions/452969/does-python-have-an-equivalent-to-java-class-forname
def get_class(cls):
    parts = cls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m