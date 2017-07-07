#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import input_data

mnist = input_data.read_data_sets('MNIST/', one_hot = True)

hidden_layer_units = 500
learn_rate = 0.01
batch_size = 100
epoches = 1000

X = tf.placeholder('float', [None, 784])
Y = tf.placeholder('float', [None, 10])

W1 = tf.Variable(tf.random_normal([784, hidden_layer_units]), \
                 dtype = tf.float32, name = 'Weight1')
b1 = tf.Variable(tf.random_normal([hidden_layer_units]), \
                 dtype = tf.float32, name = 'biase1')
hidden_out = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))

W2 = tf.Variable(tf.random_normal([hidden_layer_units, 10]), \
                 dtype = tf.float32, name = 'Weight2')
b2 = tf.Variable(tf.random_normal([10]), dtype = tf.float32, name = 'biase2')
y = tf.nn.relu(tf.add(tf.matmul(hidden_out, W2), b2))

cross_entropy = -tf.reduce_mean(Y * tf.log(y))
train_op = tf.train.GradientDescentOptimizer(learn_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(epoches):
    batch = mnist.train.next_batch(batch_size)
    if i%100 ==0:
        train_accuracy = sess.run(accuracy, feed_dict = \
                                      {X: batch[0], Y: batch[1]})
        print('step%d, train_accuracy%g'%(i, train_accuracy))
    sess.run(train_op, feed_dict = {X:batch[0], Y:batch[1]})

print('test accuracy%g'%sess.run(accuracy, feed_dict = {X: mnist.test.images, \
                                                  Y: mnist.test.labels}))

