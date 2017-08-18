import tensorflow as tf
import numpy as np
sess = tf.Session()

learning_rate = 0.05
iterations = 1400

x_vals = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
y_vals = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

my_output = tf.add(x_data, A)
my_output_expand = tf.expand_dims(my_output, 0)
y_target_expand = tf.expand_dims(y_target, 0)

sess.run(tf.global_variables_initializer())
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output_expand, labels=y_target_expand)
my_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(xentropy)

for iteration in range(iterations):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(my_opt, feed_dict={x_data: rand_x, y_target: rand_y})

    if(iteration+1) % 200 == 0:
        print('Step #' + str(iteration+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={x_data: rand_x, y_target: rand_y})))
