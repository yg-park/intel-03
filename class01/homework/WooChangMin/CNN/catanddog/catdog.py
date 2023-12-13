import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds





#데이터 준비
(train_data, validation_data, test_data), metadata = tfds.load(
'cats_vs_dogs', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], 
with_info = True, as_supervised=True,)


#train_data, validation_data, test_data에 각각 전체 데이터셋의 80%, 10%, 10%으로 나눠 사용
#tfds.load 메소드는 데이터를 다운로드하여 캐시하고 tf.data.Dataset 오브젝트를 리턴






#데이터 확인
import matplotlib.pyplot as plt
#matplotlib inline
#config InlineBackend.figure_format = 'retina'

plt.figure(figsize=(10,5))#figsize를 통해 출력할 표의 크기 설정


get_label_name = metadata.features['label'].int2str#get_label_name은 metadata 안에 label 컬럼안에서 가져온다.

#tf.data.Dataset의 take 함수를 사용해 데이터를 10개 가져온다.
for idx, (image, label) in enumerate(train_data.take(10)):
    plt.subplot(2, 5, idx+1)
    plt.imshow(image)
    plt.title(f'label {label} : {get_label_name(label)}')
    plt.axis('off')

#이미지 포맷

IMG_SIZE = 160
#사진 크기가 제각각이라 format_example() 함수로 이미지 사이즈를 모두 통일해준다.
def format_example(image, label):
    image = tf.cast(image, tf.float32)	
    image = (image/127.5) - 1	# 픽셀 scale 수정
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


train = train_data.map(format_example)
validation = validation_data.map(format_example)
test = test_data.map(format_example)



plt.figure(figsize=(10, 5))

get_label_name = metadata.features['label'].int2str

for idx, (image, label) in enumerate(train.take(10)):
    plt.subplot(2, 5, idx+1)
    image = (image + 1) / 2
    plt.imshow(image)
    plt.title(f'label {label} : {get_label_name(label)}')
    plt.axis('off')

#모델 준비
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential([
    Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(160, 160, 3)),
    MaxPooling2D(),
    Conv2D(filters=20, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(units=512, activation='relu'),
    Dense(units=2, activation='softmax')
    ])

model.summary()

#모델 학습
learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate), 
    loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

for image_batch, label_batch in train_batches.take(1):
    pass
    
image_batch.shape, label_batch.shape

#모델학습 전 테스트
validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

print("initial loss : {:.2f}".format(loss0))
print("initial accuracy : {:.2f}".format(accuracy0))


EPOCHS = 4
history = model.fit(train_batches, epochs=EPOCHS, validation_data=validation_batches)


#모델 학습 결과 확인
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

#모델 예측

for image_batch, label_batch in test_batches.take(1):
    images = image_batch
    labels = label_batch
    predictions = model.predict(image_batch)
    pass
    plt.figure(figsize=(20, 12))

for idx, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
    plt.subplot(4, 8, idx+1)
    image = (image + 1) / 2
    plt.imshow(image)
    correct = label == prediction
    title = f'real: {label} / pred : {prediction} \n {correct}!'
    if not correct:
        plt.title(title, fontdict={'color' : 'red'})
    else:
        plt.title(title, fontdict={'color' : 'blue'})
    plt.axis('off')

count = 0
for image, label, prediction in zip(images, labels, predictions):
    correct = label == prediction
    if correct:
        count += 1

print(count)


