import tensorflow as tf
sess = tf.Session()

#method 1
hypothesis = list('bearbeer')
truth = list('beersbeers')
h1 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,1,0], [0,1,1],\
                      [0,1,2], [0,1,3]], hypothesis, [1,2,4])
t1 = tf.SparseTensor([[0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,1,0],\
                      [0,1,1], [0,1,2], [0,1,3], [0,1,4]], truth, [1,2,5])

print(sess.run(tf.edit_distance(h1, t1, normalize=True)))

#method 2
hypothesis_words = ['bear', 'bar', 'tensor', 'flow']
truth_word = ['beers']
num_h_words = len(hypothesis)
h_indices = [[xi, 0, yi] for xi,x in enumerate(hypothesis_words) for yi,y in enumerate(x)]
h_chars = list(''.join(hypothesis_words))
h3 = tf.SparseTensor(h_indices, h_chars, [num_h_words, 1, 1])
truth_word_vec = truth_word*num_h_words
t_indices = [[xi, 0, yi] for xi,x in enumerate(truth_word_vec) for yi,y in enumerate(x)]
t_chars = list(''.join(truth_word_vec))
t3 = tf.SparseTensor(t_indices, t_chars, [num_h_words,1,1])

print(sess.run(tf.edit_distance(h3, t3, normalize=True)))
