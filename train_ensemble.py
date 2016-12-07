import tensorflow as tf 
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os

# Use PrettyTensor to simplify Neural Network construction.
import prettytensor as pt


from tensorflow.examples.tutorials.mnist import input_data
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


"""x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_pred = tf.nn.softmax(tf.matmul(x, W) + b)
y_true = tf.placeholder(tf.float32, [None, 10])
y_true_cls = tf.argmax(y_true, dimension=1)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y_pred), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))"""


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
							strides = [1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 2048])
b_fc1 = bias_variable([2048])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([2048, 10])
b_fc2 = bias_variable([10])


probs = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


temperature = 1
t_probs = tf.nn.softmax(tf.div(probs, temperature))
y_conv = tf.nn.softmax(probs)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(probs, y_))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def get_save_path(net_number):
	return save_dir + 'network' + str(net_number)
def init_variables():
	session.run(tf.initialize_all_variables())

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

def optimize(num_iterations, x_train, y_train, session):
	# Start-time used for printing time-usage below.
	start_time = time.time()

	for i in range(num_iterations):

		# Get a batch of training examples.
		# x_batch now holds a batch of images and
		# y_true_batch are the true labels for those images.
		x_batch, y_true_batch = random_batch(x_train, y_train)

		# Put the batch into a dict with the proper names
		# for placeholder variables in the TensorFlow graph.
		feed_dict_train = {x: x_batch,
						   y_: y_true_batch, keep_prob:.5}

		# Run the optimizer using this batch of training data.
		# TensorFlow assigns the variables in feed_dict_train
		# to the placeholder variables and then runs the optimizer.
		session_probs = session.run([optimizer, probs], feed_dict=feed_dict_train)


		# Print status every 100 iterations and after last iteration.
		if i % 100 == 0:

			# Calculate the accuracy on the training-batch.
			acc = session.run(accuracy, feed_dict=feed_dict_train)
			
			# Status-message for printing.
			msg = "Optimization Iteration: {0:>6}, Training Batch Accuracy: {1:>6.1%}"

			# Print it.
			print(msg.format(i + 1, acc))

	# Ending time.
	end_time = time.time()

	# Difference between start and end-times.
	time_dif = end_time - start_time

	# Print the time-usage.
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))



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
        feed_dict = {x: images[i:j, :], keep_prob:1.0}

        # Calculate the predicted labels using TensorFlow.
        pred_labels[i:j] = session.run(y_conv, feed_dict=feed_dict)

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
        feed_dict = {x: images[i:j, :], keep_prob:1.0}

        # Calculate the predicted labels using TensorFlow.
        pred_labels[i:j] = session.run(probs, feed_dict=feed_dict)

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
        saver.restore(sess=session, save_path=get_save_path(i))

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


session = tf.Session()
def main(): 
	
	saver = tf.train.Saver(max_to_keep=100)
	save_dir = 'checkpoints/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	

	
	
	num_networks = 3
	
	
	num_iterations = 100000
	done_training = True
	while not done_training:
	# For each of the neural networks.
		for i in range(num_networks):
			print("Neural network: {0}".format(i))

		# Create a random training-set. Ignore the validation-set.
			x_train, y_train, _, _ = random_training_set()

		# Initialize the variables of the TensorFlow graph.
			session.run(tf.initialize_all_variables())

		# Optimize the variables using this training-set.
			optimize(num_iterations=num_iterations,
				 x_train=x_train,
				 y_train=y_train, session=session)

		# Save the optimized variables to disk.
			saver.save(sess=session, save_path=get_save_path(i))

		# Print newline.
			print()
			if i == num_networks - 1:
				done_training = True 
	test_ensemble(num_networks, saver)
	x_train, y_train, _, _ = random_training_set()
	prob_labels = ensemble_probs(num_networks, saver, x_train)
	avg_prob_labels = np.zeros([y_train.shape[0], y_train.shape[1]])
	for i in range(num_networks):
		avg_prob_labels += prob_labels[i]
	avg_prob_labels = avg_prob_labels / num_networks

	np.save("avg_prob_labels.npy", avg_prob_labels)
	np.save("x_train.npy", x_train)
	np.save("y_train.npy", y_train)		

	

def test_ensemble(num_networks, saver):
	pred_labels, test_accuracies, val_accuracies = ensemble_predictions(num_networks, saver)
	print("Mean test-set accuracy: {0:.4f}".format(np.mean(test_accuracies)))
	print("Min test-set accuracy:  {0:.4f}".format(np.min(test_accuracies)))
	print("Max test-set accuracy:  {0:.4f}".format(np.max(test_accuracies)))
	ensemble_pred_labels = np.mean(pred_labels, axis=0)
	ensemble_pred_labels.shape
	ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)
	ensemble_cls_pred.shape
	ensemble_correct = (ensemble_cls_pred == data.test.cls)
	ensemble_incorrect = np.logical_not(ensemble_correct)
	best_net = np.argmax(test_accuracies)
	best_net_pred_labels = pred_labels[best_net, :, :]
	best_net_cls_pred = np.argmax(best_net_pred_labels, axis=1)
	best_net_correct = (best_net_cls_pred == data.test.cls)
	best_net_incorrect = np.logical_not(best_net_correct)

	print np.sum(ensemble_correct)
	print np.sum(best_net_correct)


if __name__ == '__main__':
	main()
