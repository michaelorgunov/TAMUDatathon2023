import tensorflow as tf
import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_HOME'] = 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11' # CHANGE THIS TO YOUR CUDA LOCATION
                                                                                       # CUDA SHOULD BE AUTOMATICALLY INSTALLED

# List available devices
devices = tf.config.experimental.list_physical_devices()

# Print the list of devices
for device in devices:
    print(device)