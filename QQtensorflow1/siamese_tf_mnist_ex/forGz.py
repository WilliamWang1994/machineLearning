# from tensorflow.examples.tutorials.mnist import input_data,mnist
from tensorflow.contrib.learn.python.learn.datasets import mnist
import gzip
import numpy
with open('image.gz', 'rb') as f:
 # out = mnist.extract_images(f)
 with gzip.GzipFile(fileobj=f) as bytestream:
     magic = mnist._read32(bytestream)
     # if magic != 2051:
     #     raise ValueError('Invalid magic number %d in MNIST image file: %s' %
     #                      (magic, f.name))
     num_images = mnist._read32(bytestream)
     rows = mnist._read32(bytestream)
     cols = mnist._read32(bytestream)
     buf = bytestream.read(rows * cols * num_images)
     data = numpy.frombuffer(buf, dtype=numpy.uint8)
     data = data.reshape(num_images, rows, cols, 1)
print('ok')
print(data)