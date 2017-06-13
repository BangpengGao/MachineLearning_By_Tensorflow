# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

learn_rate = 0.005
train_epoch = 500
batch_size = 10

x = tf.placeholder(shape = [None, 1], dtype = tf.float32)
y = tf.placeholder(shape = [None, 1], dtype = tf.float32)

w = tf.Variable(tf.random_normal([1,1]), name='weights')
b = tf.Variable(tf.random_normal([1,1]), name='baise')

y_ = tf.add(tf.matmul(x, w), b)

cost = tf.reduce_mean(tf.pow(y - y_, 2))

train_op = tf.train.GradientDesentOptimizer(learn_rate).minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(train_epoch):
    index = np.random.choice(len(data_x),size = batch_size)
    x_train = data_x[index], y_train = data_y[index]
    sess.run(train_op, feed_dict = {x: x_train, y: y_train})
    loss = sess.run(cost, feed_dict = {x: x_train, y: y_train})
    if i % 50 == 0:
        print('Step #'+str(i)+'w = '+str(sess.run(w))+' b = '+str(sess.run(b)))
        print(' cost = '+str(loss))

