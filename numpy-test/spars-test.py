import tensorflow as tf
import numpy as np
import pandas as pd

eye_color = tf.contrib.layers.sparse_column_with_keys(
  column_name="eye_color", keys=["blue", "brown", "green"])

print(eye_color)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # sess.run(init)
    output = sess.run(eye_color)
    print(output)