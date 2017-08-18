import tensorflow as tf
import matplotlib.pyplot as plt
sess = tf.Session()

#using for regression
x_vals = tf.linspace(-1.0, 1.0, 500)
target = tf.constant(0.0)

#L2 norm have a api tf.nn.l2_loss(), but it is half of L2 norm
def L2_norm_Loss(target, x_vals):
    l2_y_vals = tf.square(target - x_vals)
    l2_y_out = sess.run(l2_y_vals)
    return(l2_y_out)

def L1_norm_Loss(target, x_vals):
    l1_y_vals = tf.abs(target - x_vals)
    l1_y_out = sess.run(l1_y_vals)
    return(l1_y_out)

def Pseudo_Huber_Loss(deltas, target, x_vals):
    delta = tf.constant(deltas)
    phuber_y_vals = tf.multiply(tf.square(delta), tf.sqrt(1.0 + tf.square((target - x_vals) / delta)) - 1.0)
    phuber_y_out = sess.run(phuber_y_vals)
    return(phuber_y_out)

#using for classfing
x_vals1 = tf.linspace(-3., 5., 500)
target1 = tf.constant(1.)
targets1 = tf.fill([500, ], 1.)

def Hinge_loss(target, x_vals):
    hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
    hinge_y_out = sess.run(hinge_y_vals)
    return(hinge_y_out)

def Cross_entropy_loss(target, x_vals):
    xentropy_y_vals = - tf.multiply(target, tf.log(x_vals)) - tf.multiply((1. - target), tf.log(1. - x_vals))
    xentropy_y_out = sess.run(xentropy_y_vals)
    return(xentropy_y_out)

def Sigmoid_cross_entropy_loss(targets, x_vals):
    xentropy_sigmoid_y_vals = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=x_vals)
    xentropy_sigmoid_y_out = sess.run(xentropy_sigmoid_y_vals)
    return(xentropy_sigmoid_y_out)

def Weighted_cross_entropy(weight, targets, x_vals):
    weights = tf.constant(weight)
    xentropy_weight_y_vals = tf.nn.weighted_cross_entropy_with_logits(targets, x_vals, weights)
    xentropy_weight_y_out = sess.run(xentropy_weight_y_vals)
    return(xentropy_weight_y_out)

'''
tf.nn.softmax_cross_entropy_with_logits()
运行于非正则的输出上。这个函数被用于测量仅仅有一个类的损失。
基于此，这个函数把输出转化成概率分布。然后从真实的概率分布计算损失函数
'''

'''
tf.nn.sparse_softmax_cross_entropy_with_logits()
it is same as previously, except instead of the target being a probability distribution, it is an
index of which category is true. Instead of a sparse all-zero target vector with one value of one,
we just pass in the index of which category is the true value.
'''

#use matplotlib to plot the regression loss functions
def matplotlib_plot_regression():
    x_array = sess.run(x_vals)
    plt.plot(x_array, L2_norm_Loss(target, x_vals), 'b-', label='L2 Loss')
    plt.plot(x_array, L1_norm_Loss(target, x_vals), 'r--', label='L1 Loss')
    plt.plot(x_array, Pseudo_Huber_Loss(0.25, target, x_vals), 'k-.', label='P-Huber Loss(0.25)')
    plt.plot(x_array, Pseudo_Huber_Loss(5.0, target, x_vals), 'g:', label='P-Huber Loss(5)')
    plt.ylim(-0.2, 0.4)
    plt.legend(loc='lower right', prop={'size': 11})
    plt.show()

#use matplotlib to plot the various classification loss functions
def matplotlib_plot_classification():
    x_array = sess.run(x_vals1)
    plt.plot(x_array, Hinge_loss(target1, x_vals1), 'b-', label='Hinge Loss')
    plt.plot(x_array, Cross_entropy_loss(target1, x_vals1), 'r--', label='Cross entropy Loss')
    plt.plot(x_array, Weighted_cross_entropy(0.5, targets1, x_vals1), 'k-.', label='Weighted cross entropy')
    plt.plot(x_array, Sigmoid_cross_entropy_loss(targets1, x_vals1), 'g:', label='Sigmoid cross entropy Loss')
    plt.ylim(-1.5, 3)
    plt.legend(loc='lower right', prop={'size': 11})
    plt.show()

matplotlib_plot_classification()
