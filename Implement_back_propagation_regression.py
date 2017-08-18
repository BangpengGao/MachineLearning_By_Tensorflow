import numpy as np
import tensorflow as tf

sess = tf.Session()
learning_rate = 0.02
iterations = 100

x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1]))

my_output = tf.multiply(x_data, A)
loss = tf.square(my_output - y_target)

sess.run(tf.global_variables_initializer())
my_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

for iteration in range(iterations):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(my_opt, feed_dict={x_data: rand_x, y_target: rand_y})
    if(iteration+1)%25 == 0:
        print('Step #' + str(iteration) + 'A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))
