import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow.keras import layers

from tensorflow.keras import layers

import logging
longger = tf.get_logger()
logger.setLevel(logging.ERROR)

(training_set, validation_set), dataset_info = tfds.load('tf_flowers',split=['reain[:70%]','train[70%:]'],with_info=True,as_supervised=True,)

num_classes = dataset_info.features['label'].num_classes

num_training_examples = 0

