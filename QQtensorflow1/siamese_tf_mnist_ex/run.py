
""" Siamese implementation using Tensorflow with MNIST example.
This siamese network embeds a 28x28 image (a point in 784D) 
into a point in 2D.

By Youngwook Paul Kwon (young at berkeley.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import system things
from tensorflow.examples.tutorials.mnist import input_data # for data
import tensorflow as tf
import numpy as np
import os
from PIL import Image

import inference
import visualize


# prepare data and tf.session
models_dir = "E:/machineLearning/QQtensorflow1/siamese_tf_mnist_ex/train_models"

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
# print(mnist)
sess = tf.InteractiveSession()

# setup siamese network
siamese = inference.siamese()
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
saver = tf.train.Saver()
tf.initialize_all_variables().run()

# if you just want to load a previously trainmodel?
new = True
model_ckpt = 'E:/machineLearning/QQtensorflow1/siamese_tf_mnist_ex/train_models/model.ckpt'
if os.path.isfile(model_ckpt):
    print('1212312321')
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model.ckpt file. Do you want to load it [yes/no]?")
    if input_var == 'yes':
        new = False
print('new',new)
# check_point_path = os.path.join(models_dir, model_ckpt)
# print(check_point_path)
# ckpt = tf.train.get_checkpoint_state(models_dir)
# if ckpt and ckpt.model_checkpoint_path:
#     new = False
# print('new',new)
# start training
# embed = None
if new:
    for step in range(1001):
        batch_x1, batch_y1 = mnist.train.next_batch(128)
        batch_x2, batch_y2 = mnist.train.next_batch(128)
        batch_y = (batch_y1 == batch_y2).astype('float')

        # print(type(batch_x1))
        _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                            siamese.x1: batch_x1,
                            siamese.x2: batch_x2,
                            siamese.y_: batch_y})

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()

        if step % 100 == 0:
            print('step %d: loss %.3f' % (step, loss_v))

        if step % 1000 == 0 and step > 0:
            saver.save(sess, model_ckpt)
            embed = siamese.o1.eval({siamese.x1: mnist.test.images})
            print('embed:',embed)
            embed.tofile('embed.txt')
# else:
#     saver.restore(sess, ckpt.model_checkpoint_path)/
#     image_x1 = np.reshape(Image.open('1.jpg'), (1, 784))
#     image_x2 = np.reshape(Image.open('2.jpg'), (1, 784))
#
#     # image_x1, image_x2 = sess.run([image_x1, image_x2])
#
#     _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
#         siamese.x1: image_x1,
#         siamese.x2: image_x2,
#         })
    # add myself
    # pass

# visualize result
# mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
x_test = mnist.test.images
x_test = x_test.reshape([-1, 28, 28])

embed = np.fromfile('embed.txt', dtype=np.float32)
embed = embed.reshape([-1, 2])
visualize.visualize(embed, x_test)
# x_test = mnist.test.images.reshape([-1, 28, 28])
# visualize




