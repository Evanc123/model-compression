# python imports
import tensorflow as tf
import getopt
import sys

# custom imports
from utils import get_class
import constants as c

# model imports
from test_model import TestModel

# TODO (All): What's a good way of keeping track of the hyperparameters on each model run?

def usage():
    print 'Options:'
    print '-m/--model_type=     <Module.ClassName> (Default: test_model.TestModel)'
    print '-l/--load_path=      <Relative/path/to/saved/model>'
    print '-t/--train_dir=      <Directory/of/train/data/>'
    print '-v/--validation_dir= <Directory/of/validation/data/>'
    print '-n/--name=           <Subdirectory of ../Data/Save/*/ in which to save output>'
    print '-O/--overwrite       (Overwrites all previous data for the model with this save name)'
    print '-T/--test_only       (Only runs a test step -- no training)'
    print '-C/--write_csv       (Writes a CSV after testing)'
    print '--stats_freq=        <How often to print loss/train error stats, in # steps>'
    print '--summary_freq=      <How often to save loss/error summaries, in # steps>'
    print '--test_freq=         <How often to test the model on test data, in # steps>'
    print '--model_save_freq=   <How often to save the model, in # steps>'

def main():
    ##
    # Handle cmd line input
    ##

    model_type = 'test_model.TestModel'
    model_load_path = None
    test_only = False
    write_csv = False

    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'm:l:t:v:n:OT',
                                ['model_type=', 'load_path=', 'train_dir=', 'validation_dir='
                                 'name=', 'overwrite', 'test_only', 'write_csv', 'stats_freq=',
                                 'summary_freq=', 'test_freq=', 'model_save_freq='])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-m', '--model_type'):
            model_type = arg
        if opt in ('-l', '--load_path'):
            model_load_path = arg
        if opt in ('-t', '--train_dir'):
            c.TRAIN_DIR = arg
        if opt in ('-v', '--validation_dir'):
            c.TEST_DIR = arg
        if opt in ('-n', '--name'):
            c.set_save_name(arg)
        if opt in ('-O', '--overwrite'):
            c.clear_save_name()
        if opt in ('-T', '--test_only'):
            test_only = True
        if opt in ('-C', '--write_csv'):
            write_csv = True
        if opt == '--stats_freq':
            c.STATS_FREQ = int(arg)
        if opt == '--summary_freq':
            c.SUMMARY_FREQ = int(arg)
        if opt == '--test_freq':
            c.TEST_FREQ = int(arg)
        if opt == '--model_save_freq':
            c.MODEL_SAVE_FREQ = int(arg)

    ##
    # Run the model
    ##

    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter(c.SUMMARY_SAVE_DIR, graph=sess.graph)

    print 'Init model...'
    model = get_class(model_type)()

    print 'Init variables...'
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    # if load path specified, load a saved model
    if model_load_path is not None:
        saver.restore(sess, model_load_path)
        print 'Model restored from ' + model_load_path

    if test_only:
        model.test(sess, summary_writer, write_csv=write_csv)
    else:
        model.train(0, sess, saver, summary_writer, write_csv=write_csv)

if __name__ == '__main__':
    main()