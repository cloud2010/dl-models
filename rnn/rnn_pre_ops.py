from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
# 避免输出TensorFlow未编译CPU指令集信息
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/datasets/", one_hot=True)
# x, y = mnist.train.next_batch(1)
# print("X:", x, "shape:", x.shape)
# print("Y:", y, "shape:", y.shape)
# sess = tf.Session()
# print(sess.run(mnist.train.next_batch(10)))

'''
Example:
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]

dataset = tf.data.Dataset.from_tensor_slices(filenames)

# Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
# and then concatenate their contents sequentially into a single "flat" dataset.
# * Skip the first line (header row).
# * Filter out lines beginning with "#" (comments).
dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
        .skip(1)
        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))
'''

FILEPATH = '/datasets/800.csv'
df = pd.read_csv(FILEPATH)
print(df.rows[1])
