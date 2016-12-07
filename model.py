import tensorflow as tf
import csv
from os.path import join

import constants as c

# noinspection PyAttributeOutsideInit
class Model:
    def __init__(self):
        self.l_rate = 0.5

        # soft-initialize variables that will be in all models, (needed for train() and test())
        self.train_op    = None
        self.loss        = None
        self.global_step = None
        self.preds_test  = None
        self.loss_test   = None

        # initialize summaries lists that will be filled in define_graph()
        self.summaries_train = []
        self.summaries_test = []

        self.define_graph()

        # add summaries to visualize in TensorBoard
        self.summaries_train = tf.merge_summary(self.summaries_train)
        self.summaries_test = tf.merge_summary(self.summaries_test)

    def define_graph(self):
        """
        Setup the model graph in TensorFlow.
        """
        pass

    def get_preds(self, inputs):
        pass

    def train(self, sess, saver, summary_writer, write_csv=False):
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        ##
        # Run the training loop
        ##

        global_step = 0
        try:
            while not coord.should_stop():
                _, loss, global_step, summaries = sess.run(
                    [self.train_op, self.loss, self.global_step, self.summaries_train])

                # print train statistics
                if global_step % c.STATS_FREQ == 0:
                    print 'Step: %d | Loss: %f' % (global_step, loss)

                # save summaries
                if global_step % c.SUMMARY_FREQ == 0:
                    summary_writer.add_summary(summaries)
                    print 'Saved summaries!'

                # save the models
                if global_step % c.MODEL_SAVE_FREQ == 0:
                    print '-' * 30
                    print 'Saving models...'
                    saver.save(sess,
                               c.MODEL_SAVE_DIR + 'model.ckpt',
                               global_step=global_step)
                    print 'Saved models!'
                    print '-' * 30

                # test on validation data
                if global_step % c.TEST_FREQ == 0:
                    self.test(global_step, sess, summary_writer, write_csv=write_csv)
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (c.NUM_EPOCHS, global_step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)

    def test(self, global_step, sess, summary_writer, write_csv=False):
        def write_test_csv(angle_preds, step):
            path = join(c.TEST_SAVE_DIR, str(step) + '.csv')
            data = zip(range(len(angle_preds)), angle_preds)

            with open(path, 'wb') as f:
                writer = csv.writer(f)
                writer.writerows(data)

        ##
        # Run testing loop (should just be one batch)
        ##

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                preds, loss, summaries = sess.run(
                    [self.preds_test, self.loss_test, self.summaries_test])

                # print test statistics
                print 'Test loss: %f' % loss
                # save summaries
                summary_writer.add_summary(summaries)
                print 'Saved test summaries!'

                if write_csv:
                    write_test_csv(preds, global_step)
                    print 'Saved preds to CSV!'
        except tf.errors.OutOfRangeError:
            print 'Done testing.'
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
