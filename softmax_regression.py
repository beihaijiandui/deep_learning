import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

tmp1 = y_ * tf.log(y)
tmp2 = -tf.reduce_sum(tmp1)
cross_entropy = tf.reduce_mean(tmp2)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

batch_xs, batch_ys = mnist.train.next_batch(100)
# # y_shape =
# # y_shape = y_shape.shape
print('y', sess.run(y, feed_dict={x:batch_xs, y_:batch_ys}).shape)
print('tmp1', sess.run(tmp1, feed_dict={x:batch_xs, y_:batch_ys}).shape)
# print('tmp2', sess.run(tmp2, feed_dict={x:batch_xs, y_:batch_ys}))
# print('cross_entropy', sess.run(cross_entropy, feed_dict={x:batch_xs, y_:batch_ys}))
# xxx = 2.0
# print('xxx', tf.shape(xxx))
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})



