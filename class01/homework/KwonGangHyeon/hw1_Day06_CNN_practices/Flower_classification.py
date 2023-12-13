import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

(training_set, validation_set), dataset_info = tfds.load(
    'tf_flowers', 
    split=['train[:70%]', 'train[70%:]'],
    with_info=True,
    as_supervised=True,
)

num_classes = dataset_info.features['label'].num_classes
num_traning_examples = 0
num_validation_examples = 0

for example in training_set:
    num_traning_examples += 1

for example in validation_set:
    num_validation_examples += 1

print(f'Total Number of Classes: {num_classes}')
print(f'Total Number of Training Images: {num_traning_examples}')
print(f'Total Number of Validation Images: {num_validation_examples}\n')

for i, example in enumerate(training_set.take(5)):
    print(f'Image {i+1} shape: {example[0].shape} label: {example[1]}')

IMAGE_RES = 224

def format_image(image, label):
    image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES)) / 255.0
    return image, label

BATCH_SIZE = 32

train_batches = training_set\
    .shuffle(num_traning_examples // 4)\
        .map(format_image)\
            .batch(BATCH_SIZE)\
                .prefetch(1)

validation_batches = validation_set\
    .map(format_image)\
        .batch(BATCH_SIZE)\
            .prefetch(1)

# 전이학습을 하기 위한 준비과정
# 전이학습: 어떤 목적을 이루기 위해 학습된 모델을 다른 작업에 이용하는 것
# URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
URL = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))

feature_extractor.trainable = False

# 제일 끝단만 학습할 것
model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(num_classes)
])

model.summary()
model.compile(
    optimizer='adam', 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    metrics=['accuracy']
)

EPOCHS = 6

history = model.fit(train_batches, epochs=EPOCHS,validation_data=validation_batches)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epoches_range = range(EPOCHS)

# """
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epoches_range, acc, label='Training Accuracy')
plt.plot(epoches_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epoches_range, loss, label='Training Loss')
plt.plot(epoches_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# """

class_names = np.array(dataset_info. features['label'].names)

print(class_names)

image_batch, label_batch = next(iter(train_batches))

image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch)

predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]

print(predicted_class_names)

print('Labels: ', label_batch)
print('Predicted labels: ', predicted_ids)

plt.figure(figsize=(10, 9))
for i in range(30):
    plt.subplot(6, 5, i + 1)
    plt.subplots_adjust(hspace=0.3)
    plt.imshow(image_batch[i])
    color = 'blue' if predicted_ids[i] == label_batch[i] else 'red'
    plt.title(predicted_class_names[i].title(), color=color)
    plt.axis('off')
    plt.suptitle('Model predictions (blue: correct, red: incorrect)')

plt.show()

# https://iotnbigdata.tistory.com/24
