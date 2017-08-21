import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.Session()

learning_rate = 0.02
iterations = 100
batch_size = 20

def Batch_loss():
    x_vals = np.random.normal(1., 0.1, 100)
    y_vals = np.repeat(10., 100)
    x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[1, 1]))

    my_output = tf.multiply(x_data, A)
    loss = tf.reduce_mean(tf.square(my_output - y_target))
    my_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    #batch train
    sess.run(tf.global_variables_initializer())
    loss_batch = []
    for iteration in range(iterations):
        rand_index = np.random.choice(100, size=batch_size)
        rand_x = np.transpose([x_vals[rand_index]])
        rand_y = np.transpose([y_vals[rand_index]])
        sess.run(my_opt, feed_dict={x_data: rand_x, y_target: rand_y})

        if(iteration+1) % 5 == 0:
            print("Step #" + str(iteration+1) + " A = " + str(sess.run(A)))
            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
            print("Loss = " + str(temp_loss))
            loss_batch.append(temp_loss)
    return(loss_batch)

#stochastic train
def Stochastic_loss():
    x_vals = np.random.normal(1., 0.1, 100)
    y_vals = np.repeat(10., 100)

    x_data = tf.placeholder(shape=[1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[1], dtype=tf.float32)
    A = tf.Variable(tf.random_normal(shape=[1]))

    my_output = tf.multiply(x_data, A)
    loss = tf.reduce_mean(tf.square(my_output - y_target))
    my_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    sess.run(tf.global_variables_initializer())
    loss_stochastic = []
    for iteration in range(iterations):
        rand_index = np.random.choice(100)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]
        sess.run(my_opt, feed_dict={x_data: rand_x, y_target: rand_y})

        if(iteration+1) % 5 == 0:
            print("Step #" + str(iteration+1) + " A = " + str(sess.run(A)))
            temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
            print("Loss = " + str(temp_loss))
            loss_stochastic.append(temp_loss)
    return(loss_stochastic)

#visual the two loss
loss_batch = Batch_loss()
loss_stochastic = Stochastic_loss()
plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label='Stochastic Loss')
plt.plot(range(0, 100, 5), loss_batch, 'r--', label='Batch Loss, size=20')
plt.legend(loc='upper right', prop={'size': 11})
plt.show()

