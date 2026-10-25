import platform
import sys
import dvc
import matplotlib
import numpy
import pandas
import PIL
import tensorflow as tf

print("=" * 50)
print("VisionGuard environment check")
print("=" * 50)
print(f"OS          : {platform.system()} {platform.release()}")
print(f"Python      : {sys.version.split()[0]}")
print(f"Interpreter : {sys.executable}")
print(f"TensorFlow  : {tf.__version__}")
print(f"Keras       : {tf.keras.__version__}")
print(f"NumPy       : {numpy.__version__}")
print(f"pandas      : {pandas.__version__}")
print(f"Pillow      : {PIL.__version__}")
print(f"matplotlib  : {matplotlib.__version__}")
print(f"DVC         : {dvc.__version__}")
print("-" * 50)

gpus = tf.config.list_physical_devices("GPU")
print(f"Built with CUDA : {tf.test.is_built_with_cuda()}")
print(f"GPUs found      : {len(gpus)}")
for g in gpus:
    print("   ", g)
if not gpus:
    print("No GPU visible -> training will run on CPU.")