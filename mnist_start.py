#coding:utf_8

from tensorflow.examples.tutorials.mnist import input_data
import os
import scipy.misc
import numpy as np

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# print (mnist.train.images.shape)
# print (mnist.train.labels.shape)

save_dir = './MNIST_data/raw/'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

for i in range(20):
    one_hot_label = mnist.train.labels[i, :]
    label = np.argmax(one_hot_label)
    print 'mnist_train_%d.jpg label: %d'%(i, label)
    image_array = mnist.train.images[i, :]
    image_array = image_array.reshape(28, 28)
    filename = save_dir + 'mnist_train_%d.jpg'%i
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)