#coding=utf-8

import numpy as np
import tensorflow as tf

x_data = np.matrix([
                #    Users    / Movies        /     Movie Ratings  / Time /  Last Movies Rated
                #    A  B  C  /TI NH  SW  ST  / TI   NH   SW   ST  /      / TI NH  SW  ST
                    [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0],
                    [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0],
                    [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0],
                    [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0],
                    [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0],
                    [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0],
                    [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0]
                   ])

print ('x_data', x_data.shape)

y_data = np.array([5, 3, 1, 4, 5, 1, 5])

print ('y_data', y_data.shape)

# y_data.shape += (1, )
y_data = y_data.reshape([7,1])

print ('y_data', y_data.shape)

print '=============================================================================='

n, p =x_data.shape
# number of latent factors
k = 5
X = tf.placeholder('float', shape=[n, p])
# target
y = tf.placeholder('float', shape=[n, 1])

#bias and weight
w0 = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.zeros([p]))

# 正态分布，标准差
V = tf.Variable(tf.random_normal([k, p], stddev=0.01))
y_hat = tf.Variable(tf.zeros([n, 1]))

#线性部分
linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W, X), 1, keep_dims=True))
#特征交叉项
interactions = tf.multiply(0.5,
                           tf.reduce_sum(tf.subtract(tf.pow(tf.matmul(X, tf.transpose(V)), 2), tf.matmul(tf.pow(X, 2),
                                                                                                         tf.transpose(tf.pow(V, 2)))), 1, keep_dims=True))
y_hat = tf.add(linear_terms, interactions)

lambda_w = tf.constant(0.001, name='lambda_w')
lambda_v = tf.constant(0.001, name='lambda_v')

l2_norm = tf.reduce_sum(tf.add(tf.multiply(lambda_w, tf.pow(W, 2)), tf.multiply(lambda_v, tf.pow(V, 2))))
error = tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))
loss = tf.add(error, l2_norm)

eta = tf.constant(0.1)
optimizer = tf.train.AdagradOptimizer(eta).minimize(loss)

N_EPOCHS = 1000

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(N_EPOCHS):
        #样本数量
        indices = np.arange(n)
        np.random.shuffle(indices)
        x_data, y_data = x_data[indices], y_data[indices]
        sess.run(optimizer, feed_dict={X:x_data, y:y_data})

    print('MSE: ', sess.run(error, feed_dict={X:x_data,y:y_data}))
    print('Loss (regularized error):', sess.run(loss, feed_dict={X:x_data, y:y_data}))
    print('Predictions:', sess.run(y_hat, feed_dict={X:x_data, y:y_data}))
    print('Learnt weights:', sess.run(W, feed_dict={X:x_data, y:y_data}))
    print('Learnt factors:', sess.run(V, feed_dict={X:x_data, y:y_data}))





