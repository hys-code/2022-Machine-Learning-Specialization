import tensorflow as tf
print("GPU devices:", tf.config.list_physical_devices('GPU'))

a = tf.random.normal([3000, 3000])
b = tf.random.normal([3000, 3000])

with tf.device('/GPU:0'):
    c = tf.matmul(a, b)

print("Matrix multiplication successful on GPU.")
