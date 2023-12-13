import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

(train_data, validation_data, test_data), metadata = tfds.load('cats_vs_dogs', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info = True, as_supervised=True,)

print(train_data)
print(validation_data)
print(test_data)

import matplotlib.pyplot as plt
#matplotlib inline
#config InlineBackend.figure_format = 'retina'

plt.figure(figsize=(10,5))

get_label_name = metadata.features['label'].int2str

for idx, (image, label) in enumerate(train_data.take(10)):
    plt.subplot(2, 5, idx+1)
    plt.imshow(image)
    plt.title(f'label {label} : {get_label_name(label)}')
    plt.axis('off')

IMG_SIZE = 160

def format_example(image, label):
    image = tf.cast(image, tf.float32)	
    image = (image/127.5) - 1	# 픽셀 scale 수정
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


rain = train_data.map(format_example)
validation = validation_data.map(format_example)
test = test_data.map(format_example)

print(train)
print(validation)
print(test)


