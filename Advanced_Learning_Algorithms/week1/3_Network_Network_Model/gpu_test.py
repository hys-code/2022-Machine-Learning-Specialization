import tensorflow as tf
import time

# Check for GPU
if tf.config.list_physical_devices('GPU'):
    print("✅ GPU is available!")
else:
    print("❌ No GPU detected.")

# Create large matrices
matrix_size = 4096
a = tf.random.normal([matrix_size, matrix_size])
b = tf.random.normal([matrix_size, matrix_size])

# Warm-up
tf.matmul(a, b)

# Benchmark
start = time.time()
for _ in range(10):
    tf.matmul(a, b)
end = time.time()

print(f"10 matrix multiplications took: {end - start:.2f} seconds")
