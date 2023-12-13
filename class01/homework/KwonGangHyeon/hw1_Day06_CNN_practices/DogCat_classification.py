import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# 텐서플로우 데이터셋에서 훈련용 데이터 로드
(train_data, validation_data, test_data), metadata = tfds.load(
    'cats_vs_dogs', 
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

print(train_data)
print(validation_data)
print(test_data)

# 가져온 데이터가 어떤 것들인지 눈으로 확인
# %mapplotlib inline
# %config InlineBackend.figure_format = 'retina'
plt.figure(figsize=(10, 5))
get_label_name = metadata.features['label'].int2str

for idx, (image, label) in enumerate(train_data.take(10)): # 10개만 가져와서 보겠다
    plt.subplot(2, 5, idx+1)
    plt.imshow(image)
    plt.title(f'label {label} : {get_label_name(label)}')
    plt.axis('off')

# 데이터셋의 사진 크기가 제각각이라서 이미지 사이즈를 모두 통일해주는 작업이 필요하다.
IMG_SIZE = 160

def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = train_data.map(format_image)
validation = validation_data.map(format_image)
test = test_data.map(format_image)

print(train)
print(validation)
print(test)

# 최종 이미지 데이터 확인
plt.figure(figsize=(10, 5))
get_label_name = metadata.features['label'].int2str

for idx, (image, label) in enumerate(train.take(10)): 
    plt.subplot(2, 5, idx+1)
    image = (image + 1) / 2 # 픽셀값의 레인지가 ~1 ~ 1 이던것을 0 ~ 1로 바꿔줌
    plt.imshow(image)
    plt.title(f'label {label} : {get_label_name(label)}')
    plt.axis('off')


# 직접 모델 만들기
model = Sequential([
    Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(160, 160, 3)),
    MaxPooling2D(),
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(units=512, activation='relu'),
    Dense(units=2, activation='softmax'),
])

# 모델 요약 print()
model.summary()

learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

for image_batch, label_batch in train_batches.take(1):
    pass

image_batch.shape, label_batch.shape

validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

print(f'initial loss: {loss0:.2f}')
print(f'initial accuracy: {accuracy0:.2f}')

EPOCHS = 10
history = model.fit(train_batches, epochs=EPOCHS, validation_data=validation_batches)

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

for image_batch, label_batch in test_batches.take(1):
    images = image_batch
    labels = label_batch
    predictions = model.predict(image_batch)
    pass

predictions = np.argmax(predictions, axis=1)
predictions

plt.figure(figsize=(20, 12))

for idx, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
    plt.subplot(4, 8, idx + 1)
    image = (image + 1) / 2
    plt.imshow(image)
    condition = (label == prediction)
    title = f'real: {label} / pred: {prediction} \n {condition}!'
    if condition:
        plt.title(title, fontdict={'color':'blue'})
    else:
        plt.title(title, fontdict={'color':'red'})

    plt.axis('off')

count = 0
for image, label, prediction in zip(images, labels, predictions):
    condition = (label == prediction)
    if condition:
        count += 1

print(count)    
