"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

y_ = tf.placeholder(tf.float32, [None, 10])



W1 = tf.Variable(tf.zeros([784, 100]))
b1 = tf.Variable(tf.zeros([100]))
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h1, keep_prob)
W2 = tf.Variable(tf.zeros([100, 10]))
b2 = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W2) + b2)



cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(1000):
 
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob:.5})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))
"""


from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import random
import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y_ = tf.placeholder("float", [None, n_classes])

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
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

x_train_npy = np.load('x_train.npy')
y_train_npy = np.load('y_train.npy')
y_expert_npy = np.load('avg_prob_labels.npy')

def batch_gen(x, y_, y_expert_probs, batch_size):

    ran_num = random.randrange(0, len(x)-1-batch_size)

    return x[ran_num:ran_num+batch_size], \
    y_[ran_num:ran_num+batch_size], \
    y_expert_probs[ran_num:ran_num+batch_size]

def shuffle_in_unison_inplace(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]
x_train_npy, y_train_npy, y_expert_npy =  shuffle_in_unison_inplace(x_train_npy, y_train_npy, y_expert_npy)




# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    """
    for i in range(10000):

      x_batch, y_batch, y_expert_batch = batch_gen(
        x_train_npy, y_train_npy, y_expert_npy, 100)
      
      sess.run([optimizer], feed_dict={x:x_batch, y_: y_batch, keep_prob:.5})

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))
    """
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y_: batch_y,
                                                          keep_prob:.5})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob:1.0}))
