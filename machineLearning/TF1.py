import tensorflow as tf
import numpy as np
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.3+0.5

Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))
y=x_data*Weights+biases

loss=tf.reduce_mean(tf.square(y-y_data))
train=tf.train.GradientDescentOptimizer(0.5).minimize(loss)
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step%50==0:
        print(step,sess.run(Weights),sess.run(biases))