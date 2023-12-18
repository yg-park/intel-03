import tensorflow as tf


import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_datasets as tfds


from tensorflow.keras import layers


import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)




"""
fds.splits를 사용하여 이 교육 세트를 training_set 및 validation_set으로 분할
[70, 30] 분할을 수행하여 training_set에 70, validation_set에 30
"""
(training_set, validation_set), dataset_info = tfds.load(
'tf_flowers',
split=['train[:70%]', 'train[:70%]'],
with_info=True,
as_supervised=True,
)
"""
tfds.load를 사용하여 tf_flower 데이터 집합을 로드합니다.
tfds.load 함수는 필요한 모든 매개 변수를 사용하고 데이터셋에 대한 정보를 검색할 수 있도록 데이터셋 정보를 반환하는지 확인
"""

num_classes = dataset_info.features['label'].num_classes


num_training_examples = 0
num_validation_examples = 0


for example in training_set:
   num_training_examples += 1
  
for example in validation_set:
   num_validation_examples += 1
print('Total Number of Classes: {}'.format(num_classes))
print('Total Number of Training Classes: {}'.format(num_training_examples))
print('Total Number of Validation Images: {} \n'.format(num_validation_examples))


for i, example in enumerate(training_set.take(5)):
   print('Image {} shape: {} label: {}'.format(i+1, example[0].shape, example[1]))


IMAGE_RES = 299


def format_image(image, label):
   image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/255.0
   return image, label


BATCH_SIZE = 32


train_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)


validation_batches = validation_set.map(format_image).batch(BATCH_SIZE).prefetch(1)

URL = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
#URL = "http://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"# MobileNet v2를 사용하여 feature_extrator

feature_extractor = hub.KerasLayer(URL,input_shape=(IMAGE_RES, IMAGE_RES,3))
# TensorFlow Hub(최종 분류 계층이 없는)의 부분 모델을 형상 벡터라고 한다.


feature_extractor.trainable = False
#형상 추출기 계층의 변수를 동결하여 훈련은 최종 분류기 계층만 수정한다.




#사전 학습된 모델과 새 분류 계층을 추가합니다. 
#분류 계층에는 Flowers 데이터 집합과 동일한 수의 클래스가 있어야 합니다. 마지막으로 순차적 모델의 요약을 인쇄합니다
model = tf.keras.Sequential([
   feature_extractor,
   layers.Dense(num_classes)
   ])


model.summary()

#아래의 셀에서 이 모델을 다른 모델과 마찬가지로 컴파일을 호출한 다음 장착합니다. 
#두 방법을 모두 적용할 때 적절한 매개 변수를 사용해야 합니다. 모델을 6 번 동안 훈련하십시오.

model.compile(
   optimizer='adam',
   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
   metrics=['accuracy'])


EPOCHS = 6


history = model.fit(train_batches,
   epochs=EPOCHS,
   validation_data=validation_batches)





# 교육 및 검증 정확도/손실 그래프를 표시합니다.
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']


loss = history.history['loss']
val_loss = history.history['val_loss']


epochs_range = range(EPOCHS)



"""
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
"""

class_names = np.array(dataset_info.features['label'].names)


print(class_names)


image_batch, label_batch = next(iter(train_batches))


image_batch = image_batch.numpy()
label_batch = label_batch.numpy()


predicted_batch = model.predict(image_batch)
predicted_batch = tf.squeeze(predicted_batch).numpy()


predicted_ids = np.argmax(predicted_batch, axis=-1)
predicted_class_names = class_names[predicted_ids]


print(predicted_class_names)


print("Labels: ", label_batch)
print("Predicted labels: ", predicted_ids)


plt.figure(figsize=(10,9))
for n in range(30):
   plt.subplot(6,5,n+1)
   plt.subplots_adjust(hspace = 0.3)
   plt.imshow(image_batch[n])
   color = "blue" if predicted_ids[n] == label_batch[n] else "red"
   plt.title(predicted_class_names[n].title(), color=color)
   plt.axis('off')
   plt.suptitle("Model predictions (blue: correct, red: incorrect)")
  
plt.show()
