from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
mnist      =input_data.read_data_sets('data/',one_hot=True)
trainimage   =mnist.train.images
trainlabel =mnist.train.labels
testimg    =mnist.test.images
testlabel  =mnist.test.labels
print("MNIST loaded")
# print(trainimage.shape)
# print(trainlabel.shape)
# print(testimg.shape)
# print(testlabel.shape)

x=tf.placeholder("float",[None,784])
y=tf.placeholder("float",[None,10])

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

actv=tf.nn.softmax(tf.matmul(x,W)+b)

cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv),reduction_indices=1))#交叉熵

learning_rate=0.01
optm=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
pred=tf.equal(tf.argmax(actv,1),tf.argmax(y,1))
accr=tf.reduce_mean(tf.cast(pred,'float'))
init=tf.global_variables_initializer()
sess=tf.InteractiveSession()
arr=np.array([
             [31,23,4,24,27,34],
             [18, 3,25, 0, 6,35],
             [28,14,33,22,20, 8],
             [13,30,21,19, 7, 9],
             [16, 1,26,32, 2,29],
             [17,12, 5,11,10,15]
])
tf.argmax(arr,1).eval()


training_epochs=50
batch_size=100
display_step =5

sess=tf.Session()
sess.run(init)
for epoch in range(training_epochs):
    avg_cost=0
    num_batch=int(mnist.train.num_examples/batch_size)
    for i in range(num_batch):
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        sess.run(optm,feed_dict={x:batch_xs,y:batch_ys})
        feeds={x:batch_xs,y:batch_ys}
        avg_cost+=sess.run(cost,feed_dict=feeds)/num_batch

    if epoch%display_step==0:
        feeds_train={x:batch_xs,y:batch_ys}
        feeds_test={x:mnist.test.images, y:mnist.test.labels}
        train_acc=sess.run(accr,feed_dict=feeds_train)
        test_acc=sess.run(accr,feed_dict=feeds_test)
        print("Epoch:%03d/%03d cost:%.9f train_acc:%.3f test_acc:%.3f"%(epoch,training_epochs,avg_cost,train_acc,test_acc))
print("DONE")