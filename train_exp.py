import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

print ("PACKAGES LOADED")

data = input_data.read_data_sets('data/MNIST/', one_hot=True)

combined_images = np.concatenate([data.train.images, data.validation.images], axis=0)
combined_labels = np.concatenate([data.train.labels, data.validation.labels], axis=0)

combined_size = len(combined_images)
train_size = int(0.9 * combined_size)

validation_size = combined_size - train_size
data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1

# Number of classes, one class for each of 10 digits.
num_classes = 10

train_batch_size = 64

save_dir = 'checkpoints/'


def random_training_set():
	# Create a randomized index into the full / combined training-set.
	idx = np.random.permutation(combined_size)

	# Split the random index into training- and validation-sets.
	idx_train = idx[0:train_size]
	idx_validation = idx[train_size:]

	# Select the images and labels for the new training-set.
	x_train = combined_images[idx_train, :]
	y_train = combined_labels[idx_train, :]

	# Select the images and labels for the new validation-set.
	x_validation = combined_images[idx_validation, :]
	y_validation = combined_labels[idx_validation, :]

	# Return the new training- and validation-sets.
	return x_train, y_train, x_validation, y_validation


def predict_labels(images):
    # Number of images.
    num_images = len(images)

    batch_size = 256
    # Allocate an array for the predicted labels which
    # will be calculated in batches and filled into this array.
    pred_labels = np.zeros(shape=(num_images, num_classes),
                           dtype=np.float)

    # Now calculate the predicted labels for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images between index i and j.
        feed_dict = {x: images[i:j, :], keepratio:1.0}

        # Calculate the predicted labels using TensorFlow.
        pred_labels[i:j] = sess.run(_pred, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    return pred_labels


def output_pre_softmax_predictions(images):
    # Number of images.
    num_images = len(images)
    batch_size = 256
    # Allocate an array for the predicted labels which
    # will be calculated in batches and filled into this array.
    pred_labels = np.zeros(shape=(num_images, num_classes),
                           dtype=np.float)
    
    # Now calculate the predicted labels for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images between index i and j.
        feed_dict = {x: images[i:j, :], keepratio:1.0}

        # Calculate the predicted labels using TensorFlow.
        pred_labels[i:j] = sess.run(_pred, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    return pred_labels

def correct_prediction(images, labels, cls_true):
    # Calculate the predicted labels.
    pred_labels = predict_labels(images=images)

    # Calculate the predicted class-number for each image.
    cls_pred = np.argmax(pred_labels, axis=1)

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct

device_type = "/gpu:1"

n_input  = 784
n_output = 10
with tf.device(device_type):
    weights  = {
        'wc1': tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=0.1)),
        'wc2': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1)),
        'wd1': tf.Variable(tf.truncated_normal([7*7*128, 1024], stddev=0.1)),
        'wd2': tf.Variable(tf.truncated_normal([1024, n_output], stddev=0.1))
    }
    biases   = {
        'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
        'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
        'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
        'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1))
    }
    def conv_basic(_input, _w, _b, _keepratio):
        # INPUT
        _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
        # CONV LAYER 1
        _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
        _mean, _var = tf.nn.moments(_conv1, [0, 1, 2])
        _conv1 = tf.nn.batch_normalization(_conv1, _mean, _var, 0, 1, 0.0001)
        _conv1 = tf.nn.relu(tf.nn.bias_add(_conv1, _b['bc1']))
        _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)
        # CONV LAYER 2
        _conv2 = tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
        _mean, _var = tf.nn.moments(_conv2, [0, 1, 2])
        _conv2 = tf.nn.batch_normalization(_conv2, _mean, _var, 0, 1, 0.0001)
        _conv2 = tf.nn.relu(tf.nn.bias_add(_conv2, _b['bc2']))
        _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)
        # VECTORIZE
        _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])
        # FULLY CONNECTED LAYER 1
        _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
        _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)
        # FULLY CONNECTED LAYER 2
        _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])
        # RETURN
        out = { 'input_r': _input_r, 'conv1': _conv1, 'pool1': _pool1, 'pool1_dr1': _pool_dr1,
            'conv2': _conv2, 'pool2': _pool2, 'pool_dr2': _pool_dr2, 'dense1': _dense1,
            'fc1': _fc1, 'fc_dr1': _fc_dr1, 'out': _out
        }
        return out
print ("CNN READY")

# PLACEHOLDERS
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
y_expert = tf.placeholder("float", [None, 10])
keepratio = tf.placeholder(tf.float32)

# FUNCTIONS
temperature = 1
with tf.device(device_type):
    _pred = conv_basic(x, weights, biases, keepratio)['out']
    t_probs = tf.nn.softmax(tf.div(_pred, temperature))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_pred, y))
    optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    _corr = tf.equal(tf.argmax(_pred,1), tf.argmax(y,1)) 
    accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) 
    init = tf.initialize_all_variables()
    
# SAVER
save_step = 1
saver = tf.train.Saver(max_to_keep=3)
print ("GRAPH READY")



mnist = input_data.read_data_sets('data/', one_hot=True)
trainimg   = mnist.train.images
trainlabel = mnist.train.labels
trainlabelcls = np.argmax(trainlabel, axis=1)
testimg    = mnist.test.images
testlabel  = mnist.test.labels
testlabelcls = np.argmax(testlabel, axis=1)

do_train = 0
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
sess.run(init)

training_epochs = 15
batch_size      = 100
display_step    = 1
if do_train == 1:
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio:0.7})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})/total_batch

        # Display logs per epoch step
        if epoch % display_step == 0: 
            print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
            train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})
            print (" Training accuracy: %.3f" % (train_acc))
            test_acc = correct_prediction(images = testimg, labels=testlabel, cls_true = testlabelcls)
            #test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio:1.})
            print (" Test accuracy: %.3f" % (test_acc.mean()))

        # Save Net
        if epoch % save_step == 0:
            saver.save(sess, "nets/cnn_mnist_basic.ckpt-" + str(epoch))

    print ("OPTIMIZATION FINISHED")

if do_train == 0:
    epoch = training_epochs-1
    saver.restore(sess, "nets/cnn_mnist_basic.ckpt-" + str(epoch))

test_acc = correct_prediction(images = testimg, labels=testlabel, cls_true = testlabelcls)
            #test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio:1.})
print (" Final Test accuracy: %.3f" % (test_acc.mean()))




#test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio:1.})
#print (" TEST ACCURACY: %.3f" % (test_acc))




import pdb; pdb.set_trace()




        # Append the predicted labels to the list.



x_train, y_train, _, _ = random_training_set()

prob = output_pre_softmax_predictions(images=x_train)

np.save("x_train_prob_labels.npy", prob)
np.save("x_train.npy", x_train)
np.save("x_test_images.npy", y_train)		



save_dir = 'checkpoints/'





def get_save_path(net_number):
	return save_dir + 'network' + str(net_number)
def init_variables():
	sess.run(tf.initialize_all_variables())

def random_batch(x_train, y_train):
	# Total number of images in the training-set.
	num_images = len(x_train)

	# Create a random index into the training-set.
	idx = np.random.choice(num_images,
						   size=train_batch_size,
						   replace=False)

	# Use the random index to select random images and labels.
	x_batch = x_train[idx, :]  # Images.
	y_batch = y_train[idx, :]  # Labels.

	# Return the batch.
	return x_batch, y_batch




def test_correct():
    return correct_prediction(images = data.test.images,
                              labels = data.test.labels,
                              cls_true = data.test.cls)

def validation_correct():
    return correct_prediction(images = data.validation.images,
                              labels = data.validation.labels,
                              cls_true = data.validation.cls)

def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.
    return correct.mean()

def test_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the test-set.
    correct = test_correct()
    
    # Calculate the classification accuracy and return it.
    return classification_accuracy(correct)

def validation_accuracy():
    # Get the array of booleans whether the classifications are correct
    # for the validation-set.
    correct = validation_correct()
    
    # Calculate the classification accuracy and return it.
    return classification_accuracy(correct)

def ensemble_predictions(num_networks, saver):
    # Empty list of predicted labels for each of the neural networks.
    pred_labels = []

    # Classification accuracy on the test-set for each network.
    test_accuracies = []

    # Classification accuracy on the validation-set for each network.
    val_accuracies = []

    # For each neural network in the ensemble.
    for i in range(num_networks):
        # Reload the variables into the TensorFlow graph.
        saver.restore(sess=sess, save_path=get_save_path(i))

        # Calculate the classification accuracy on the test-set.
        test_acc = test_accuracy()

        # Append the classification accuracy to the list.
        test_accuracies.append(test_acc)

        # Calculate the classification accuracy on the validation-set.
        val_acc = validation_accuracy()

        # Append the classification accuracy to the list.
        val_accuracies.append(val_acc)

        # Print status message.
        msg = "Network: {0}, Accuracy on Validation-Set: {1:.4f}, Test-Set: {2:.4f}"
        print(msg.format(i, val_acc, test_acc))

        # Calculate the predicted labels for the images in the test-set.
        # This is already calculated in test_accuracy() above but
        # it is re-calculated here to keep the code a bit simpler.
        pred = predict_labels(images=data.test.images)

        # Append the predicted labels to the list.
        pred_labels.append(pred)
    
    return np.array(pred_labels), \
           np.array(test_accuracies), \
           np.array(val_accuracies)

def ensemble_probs(num_networks, saver, x_train):
    # Empty list of predicted labels for each of the neural networks.
    probs_labels = []


    # For each neural network in the ensemble.
    for i in range(num_networks):
        # Reload the variables into the TensorFlow graph.
        saver.restore(sess=session, save_path=get_save_path(i))

        prob = output_pre_softmax_predictions(images=x_train)

        # Append the predicted labels to the list.
        probs_labels.append(prob)
    
    return np.array(probs_labels)
