import tensorflow as tf

from model import Model
from utils import get_inputs, MSE_loss
import constants as c


# noinspection PyAttributeOutsideInit
class TestModel(Model):
    def __init__(self):
        self.l_rate = 0.5

        Model.__init__(self)

    def define_graph(self):
        """
        Setup the model graph in TensorFlow.
        """

        with tf.name_scope('Model'):
            with tf.name_scope('Data'):
                self.frames, self.angles = get_inputs(True, c.BATCH_SIZE, c.NUM_EPOCHS)
                self.frames_test, self.angles_test = get_inputs(False, c.NUM_VALIDATION, 1)

            with tf.name_scope('Variables'):
                with tf.name_scope('Conv'):
                    self.conv_ws = []
                    self.conv_bs = []

                    with tf.name_scope('1'):
                        self.conv_ws.append(
                            tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.01)))
                        self.conv_bs.append(tf.Variable(tf.truncated_normal([32], stddev=0.01)))

                    with tf.name_scope('2'):
                        self.conv_ws.append(
                            tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.01)))
                        self.conv_bs.append(tf.Variable(tf.truncated_normal([64], stddev=0.01)))

                with tf.name_scope('FC'):
                    self.fc_ws = []
                    self.fc_bs = []

                    with tf.name_scope('1'):
                        # TODO (Matt): Make sure these dimensions line up.
                        self.fc_ws.append(
                            tf.Variable(tf.truncated_normal([3136, 1024], stddev=0.01)))
                        self.fc_bs.append(tf.Variable(tf.truncated_normal([1024], stddev=0.01)))

                    with tf.name_scope('2'):
                        self.fc_ws.append(tf.Variable(tf.truncated_normal([1024, 1], stddev=0.01)))
                        self.fc_bs.append(tf.Variable(tf.truncated_normal([1], stddev=0.01)))

            with tf.name_scope('Training'):
                self.global_step = tf.Variable(0, trainable=False)
                self.loss = MSE_loss(self.get_preds(self.frames), self.angles)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.l_rate)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

                loss_summary = tf.scalar_summary('train_loss', self.loss)
                self.summaries_train.append(loss_summary)

            with tf.name_scope('Testing'):
                self.preds_test = self.get_preds(self.frames_test), self.angles_test
                self.loss_test = MSE_loss(self.preds_test, self.angles_test)

                loss_summary = tf.scalar_summary('test_loss', self.loss_test)
                self.summaries_test.append(loss_summary)

    def get_preds(self, inputs):
        preds = inputs
        with tf.name_scope('Calculation'):
            with tf.name_scope('Conv'):
                for layer in xrange(len(self.conv_ws)):
                    with tf.name_scope(str(layer)):
                        preds = tf.nn.conv2d(preds, self.conv_ws[layer], [1, 1, 1, 1], 'SAME')
                        preds = tf.nn.max_pool(preds, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
                        preds = tf.nn.relu(preds)

            # reshape
            shape = tf.shape(preds)
            preds = tf.reshape(preds, [shape[0], shape[1] * shape[2] * shape[3]])

            with tf.name_scope('FC'):
                with tf.name_scope('1'):
                    preds = tf.matmul(preds, self.fc_ws[0]) + self.fc_bs[0]
                    preds = tf.nn.relu(preds)
                with tf.name_scope('2'):
                    preds = tf.matmul(preds, self.fc_ws[1]) + self.fc_bs[1]

        return preds