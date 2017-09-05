# coding:utf-8
import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.misc import imsave

filename = 'MNIST_data\\train-images.idx3-ubyte'
binfile = open(filename, 'rb')
buf = binfile.read()

index = 0
# '>IIII'使用大端法读取四个unsigned int32
magic, numImages, numRows, numColumns = struct.unpack_from('>IIII', buf, index)
index += struct.calcsize('>IIII')

# 输出大端数
print(magic)
print(numImages)
print(numRows)
print(numColumns)

for i in range(60000):
    name = str(i) + ".jpg"
    # upack_from从流中截取784位数据（图片像素值）
    im = struct.unpack_from('>784B', buf, index)
    index += struct.calcsize('>784B')
    
    im = np.array(im)
    im = im.reshape(28, 28)
    imsave(name, im)

    # fig = plt.figure()
    # plotwindow = fig.add_subplot(111)
    # plt.imshow(im , cmap='gray')
    # plt.show()