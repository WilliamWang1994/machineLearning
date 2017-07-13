import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

state=tf.Variable(0)
new_value=tf.add(state,tf.constant(1))
update=tf.assign(state,new_value)
with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))


w=tf.Variable([[0.5,1.0]])
x=tf.Variable([[2.0],[1.0]])
y=tf.matmul(w,x)
init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(y.eval())

num_points=1000
vectors_set=[]
for i in range(num_points):
    x1=np.random.normal(0.0,0.55)
    y1=x1*0.1+0.3+np.random.normal(0.0,0.03)
    vectors_set.append([x1,y1])
x_data=[v[0]  for v in vectors_set]
y_data=[v[1]  for v in vectors_set]

plt.scatter(x_data,y_data,c='r')
plt.show()