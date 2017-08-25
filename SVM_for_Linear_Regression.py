import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
sess = tf.Session()

iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])
train_indices = np.random.choice(len(x_vals), \
                                 round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
y_vals_train = y_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_test = y_vals[test_indices]

batch_size = 50
learning_rate = 0.01
iterations = 1500

x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
model_output = tf.add(tf.matmul(x_data, A), b)

epsilon = tf.constant([0.5])
loss = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(\
    tf.subtract(model_output, y_target)), epsilon)))
my_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
sess.run(tf.global_variables_initializer())

train_loss = []
test_loss = []
for i in range(iterations):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(my_opt, feed_dict={x_data: rand_x, y_target: rand_y})
    train_loss.append(sess.run(loss, feed_dict={x_data: rand_x, y_target:rand_y}))
    test_loss.append(sess.run(loss, feed_dict={x_data:\
                     np.transpose([x_vals_test]), y_target:\
                     np.transpose([y_vals_test])}))

[[slope]] = sess.run(A)
[[y_intercept]] = sess.run(b)
[width] = sess.run(epsilon)

best_fit = []
best_fit_upper = []
best_fit_lower = []
for i in x_vals:
    best_fit.append(slope*i+y_intercept)
    best_fit_upper.append(slope*i+y_intercept+width)
    best_fit_lower.append(slope*i+y_intercept-width)

plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='SVM Regression Line', linewidth=3)
plt.plot(x_vals, best_fit_upper, 'r--', linewidth=2)
plt.plot(x_vals, best_fit_lower, 'r--', linewidth=2)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title("Sepal Width VS Pedal Width")
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Width')
plt.show()

plt.plot(train_loss, 'k-', label='Train set loss')
plt.plot(test_loss, 'r--', label='Test set loss')
plt.title('L2 loss per Generation')
plt.ylabel('L2 loss')
plt.xlabel('Generation')
plt.legend(loc='upper right')
plt.show()
