import os
import shutil
from datetime import datetime


##
# Data
##

def get_date_str():
    """
    @return: A string representing the current date/time that can be used as a directory name.
    """
    return str(datetime.now()).replace(' ', '_').replace(':', '.')[:-10]

def get_dir(directory):
    """
    Creates the given directory if it does not exist.

    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def clear_dir(directory):
    """
    Removes all files in the given directory.

    @param directory: The path to the directory.
    """
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(e)

# root directory for all data
DATA_DIR = get_dir('../data/')

# directory of unprocessed training frames
TRAIN_DIR = os.path.join(DATA_DIR, 'train/')
# directory of unprocessed test frames
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation/')

# the height and width of the full frames to train and test on.
FRAME_HEIGHT = 100
FRAME_WIDTH = 200

# The number of examples in the validation data
# TODO (David): Change this to be the actual number of validation examples.
NUM_VALIDATION = 100


##
# Output
##

def set_save_name(name):
    """
    Edits all constants dependent on SAVE_NAME.

    @param name: The new save name.
    """
    global SAVE_NAME, MODEL_SAVE_DIR, SUMMARY_SAVE_DIR, TEST_SAVE_DIR

    SAVE_NAME = name
    MODEL_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'models/', SAVE_NAME))
    SUMMARY_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'summaries/', SAVE_NAME))
    TEST_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'tests/', SAVE_NAME))

def clear_save_name():
    """
    Clears all saved content for SAVE_NAME.
    """
    clear_dir(MODEL_SAVE_DIR)
    clear_dir(SUMMARY_SAVE_DIR)
    clear_dir(TEST_SAVE_DIR)


# root directory for all saved content
SAVE_DIR = get_dir('../save/')

# inner directory to differentiate between runs
SAVE_NAME = 'default/'
# directory for saved models
MODEL_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'models/', SAVE_NAME))
# directory for saved TensorBoard summaries
SUMMARY_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'summaries/', SAVE_NAME))
# directory for saved test CSVs
TEST_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'tests/', SAVE_NAME))

STATS_FREQ      = 10     # how often to print loss/train error stats, in # steps
SUMMARY_FREQ    = 100    # how often to save the summaries, in # steps
TEST_FREQ       = 5000   # how often to test the model on test data, in # steps
MODEL_SAVE_FREQ = 10000  # how often to save the model, in # steps

##
# Training
##

# Set these here for consistency when testing different models. Model-specific hyperparameters (e.g.
# learning rate) should be set for individual models in their classes.
BATCH_SIZE = 32
NUM_EPOCHS = 1