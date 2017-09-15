import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import requests

sess = tf.Session()

housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE', 'DIS',\
                  'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX','PTRATIO',\
             'B', 'LSTAT']
num_features = len(cols_used)
#Request data
housing_file = requests.get(housing_url)
#Parse data
housing_data = [[float(x) for x in y.split(" ") if len(x)>=1] for y in\
                housing_file.text.split("\n") if len(y)>=1]

y_vals = np.transpose([np.array([y[13] for y in housing_data])])
x_vals = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used]\
                   for y in housing_data])

#scale x values to between 0-1
x_vals = (x_vals - x_vals.min(0)) / x_vals.ptp(0)

#create diagonal weight matrix
weight_diagonal = x_vals.std(0)
weight_matrix = tf.cast(tf.diag(weight_diagonal), dtype=tf.float32)

#split dataset into train dataset and test dataset
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_train_vals = x_vals[train_indices]
y_train_vals = y_vals[train_indices]
x_test_vals = x_vals[test_indices]
y_test_vals = y_vals[test_indices]

k = 4
batch_size = len(x_test_vals)

#declare placeholder
x_train_data = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
y_train_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x_test_data = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
y_test_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

#declare distance function
subtraction_term = tf.subtract(x_train_data, tf.expand_dims(x_test_data, 1))
first_product = tf.matmul(subtraction_term, tf.tile(tf.expand_dims(weight_matrix, 0)\
                                                    , [batch_size, 1, 1]))
second_product = tf.matmul(first_product, tf.transpose(subtraction_term, perm=[0, 2, 1]))
distance = tf.sqrt(tf.matrix_diag_part(second_product))

#return the top k-NNs, do this with the top_k() function.return the largest of
#the negative distance values
top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1), 1)
x_sums_repeated = tf.matmul(x_sums, tf.ones([1, k], tf.float32))
x_vals_weights = tf.expand_dims(tf.div(top_k_xvals, x_sums_repeated), 1)
top_k_yvals = tf.gather(y_train_target, top_k_indices)
prediction = tf.squeeze(tf.matmul(x_vals_weights, top_k_yvals), squeeze_dims=[1])

#evalute our model
mse = tf.div(tf.reduce_sum(tf.square(tf.subtract(prediction, y_test_target))), batch_size)

#loop through test and calculate the mse
num_loops = int(np.ceil(len(x_test_vals) / batch_size))
for i in range(num_loops):
    min_index = i*batch_size
    max_index = (min((i+1)*batch_size, len(x_train_vals)))
    x_batch = x_test_vals[min_index:max_index]
    y_batch = y_test_vals[min_index:max_index]
    predictions = sess.run(prediction, feed_dict={x_train_data:x_train_vals,\
                y_train_target:y_train_vals, x_test_data:x_batch, y_test_target:y_batch})
    batch_mse = sess.run(mse, feed_dict={x_train_data:x_train_vals,\
                y_train_target:y_train_vals, x_test_data:x_batch, y_test_target:y_batch})
    print('Batch #'+str(i+1)+' MSE: '+str(np.round(batch_mse, 3)))

#plot the distribution
bins = np.linspace(5, 50, 45)
plt.hist(predictions, bins, alpha=0.5, label='Prediction')
plt.hist(y_batch, bins, alpha=0.5, label='Actual')
plt.title('Histogram of Predicted and Actual Values')
plt.xlabel('Med Home Value in $1,000s')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()
