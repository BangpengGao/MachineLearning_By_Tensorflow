# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

learning_rate = 0.01
train_epoch = 500
batch_size = 10
loss = []

x = tf.placeholder("float")
y = tf.placeholder("float")

w = tf.Variable(tf.random_normal(shape), dtype = tf.float32, name = 'Weights')
b = tf.Variable(tf.random_normal(shape), dtype = tf.float32, name = 'bias')
y_ = tf.add(tf.matmul(x, w), b)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
correct_rate = tf.equal(tf,softmax(y, 1), tf,softmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_rate, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(train_epoch):
    total_batch = int(n_samples/batch_size)
    one_loss = []
    for i in range(total_batch):
        _,oneloss = sess.run([train_op, cost],\
                            feed_dict = {x:data_x[i*batch_size:(i+1)*batch_size],\
                                         y:data_y[i*batch_size:(i+1)*batch_size]})
        one_loss.append(oneloss)
    loss.append(np.mean(one_loss))

    if epoch % 50 == 0:
        print('Epoch: '+str(epoch)+" cost="+str(loss[-1]))
        print("Test_Accuracy:", accuracy.eval(feed_dict = {x:test_x, y:test_y}))
        
