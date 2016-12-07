from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import random
import tensorflow as tensorflow
sess = tf.InteractiveSession()



learning_rate = 0.001
training_epochs = 15

## 1000, 1000, dropout .5 and 10000 steps of size 100 batchces yeilds .965
# Network Parameters
n_hidden_1 = 300 # 1st layer number of features
n_hidden_2 = 300 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)


def replace_none_with_zero(l):
    return [0 if i==None else i for i in l]


x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])
y_expert = tf.placeholder("float", [None, 10])
keep_prob = tf.placeholder(tf.float32)
# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    
    layer_1_keep = tf.nn.dropout(layer_1, keep_prob)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1_keep, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2_keep = tf.nn.dropout(layer_2, keep_prob)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2_keep, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
probs = multilayer_perceptron(x, weights, biases)



"""
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



W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])


probs = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
"""

y_conv = tf.nn.softmax(probs)

temperature = 2
temp_square = temperature * temperature 


t_probs = tf.nn.softmax(tf.div(probs, temperature))
y_expert_probs = tf.nn.softmax(tf.div(y_expert, temperature))

alpha = 0

#xent = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv)))
cross_entropy = (1 - alpha) * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(probs, y_))

cross_entropy_expert = alpha * tf.reduce_mean(-tf.reduce_sum(t_probs * tf.log(y_expert_probs), reduction_indices=[1]))


optim = tf.train.AdamOptimizer(learning_rate=learning_rate)




grads_and_vars = optim.compute_gradients(cross_entropy)


temp_grads = optim.compute_gradients(cross_entropy_expert)

temp_compensated_grads = [(grad * temp_square, var) for grad, var in temp_grads]

if alpha == 0: 
    optimizer = optim.apply_gradients(grads_and_vars)
else:
    optimizer = optim.apply_gradients(grads_and_vars + temp_compensated_grads)




# Initializing the variables
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



"""
temperature = 8
temp_square = temperature * temperature 
t_probs = tf.nn.softmax(tf.div(probs, temperature))

y_expert_probs = tf.nn.softmax(tf.div(y_expert, temperature))



cross_entropy = 1 * tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
cross_entropy_expert = 0 * tf.reduce_mean(-tf.reduce_sum(y_expert_probs * tf.log(t_probs), reduction_indices=[1])) 
optim = tf.train.AdamOptimizer(0.001)





grads_and_vars = replace_none_with_zero(optim.compute_gradients(cross_entropy))

temp_grads = replace_none_with_zero(optim.compute_gradients(cross_entropy_expert))

temp_compensated_grads = [(grad * temp_square, var) for grad, var in temp_grads]


train_op = optim.apply_gradients( grads_and_vars)
"""
sess.run(tf.initialize_all_variables()) 


#sess.run(optim.apply_gradients(grads_and_vars_hard + grads_and_vars_soft))
def batch_gen(x, y_, y_expert_probs, batch_size):

    ran_num = random.randrange(0, len(x)-1-batch_size)

    return x[ran_num:ran_num+batch_size], \
    y_[ran_num:ran_num+batch_size], \
    y_expert_probs[ran_num:ran_num+batch_size]
    



x_train_npy = np.load('x_train.npy')
y_train_npy = np.load('y_test_labels.npy')
y_expert_npy = np.load('x_train_prob_labels.npy')


def shuffle_in_unison_inplace(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]
x_train_npy, y_train_npy, y_expert_npy =  shuffle_in_unison_inplace(x_train_npy, y_train_npy, y_expert_npy)


for i in range(100000):

  x_batch, y_batch, y_expert_batch = batch_gen(
    x_train_npy, y_train_npy, y_expert_npy, 100)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:x_batch, y_: y_batch, y_expert:y_expert_batch, keep_prob:1})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  if i%1000 == 0:
    print("test accuracy %g Alpha: %g Temp: %g "% (accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}), alpha, temperature))
      
  optimizer.run(feed_dict={x:x_batch, y_: y_batch, y_expert: y_expert_batch, keep_prob:.5})

print("test accuracy %g Alpha: %g Temp: %g "% (accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}), alpha, temperature))