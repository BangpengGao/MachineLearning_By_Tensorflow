import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets

sess = tf.Session()

iris = datasets.load_iris()
binary_target = np.array([1. if x==0 else 0. for x in iris.target])
iris_2d = np.array([[x[2], x[3]] for x in iris.data])

batch_size = 20
learning_rate = 0.05
iterations = 1000

x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

my_mul = tf.multiply(x2_data, A)
my_add = tf.add(my_mul, b)
my_output = tf.subtract(x1_data, my_add)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_target, logits=my_output)
my_opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(xentropy)

sess.run(tf.global_variables_initializer())

for i in range(iterations):
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    sess.run(my_opt, feed_dict={x1_data: rand_x1, x2_data: rand_x2, y_target: rand_y})

    if(i+1) % 200 == 0:
        print("Step #" + str(i+1) + " A = " + str(sess.run(A)) + ', b = ' + str(sess.run(b)))

[[slope]] = sess.run(A)
[[intercept]] = sess.run(b)
x = np.linspace(0, 3, num=50)
ablineValues = []
for i in x:
    ablineValues.append(slope*i+intercept)
setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i] == 1]
setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i] == 1]
non_setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i] == 0]
non_setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i] == 0]
plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa')
plt.plot(non_setosa_x, non_setosa_y, 'ro', label='non_setosa')
plt.plot(x, ablineValues, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 7.1])
plt.suptitle('Linear Separator For I.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()
