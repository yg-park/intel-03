#import
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#prepare dataset
mnist = tf.keras.datasets.mnist

(image_train, label_train), (image_test, label_test) = mnist.load_data()


print("Train Image shape : ", image_train.shape)
print("Train Labe : ", label_train, "\n")
print(image_train[0])

## show
"""
NUM = 20
plt.figure(figsize=(10,10))
for idx in range(NUM):
	sp = plt.subplot(5,5,idx+1)
	plt.imshow(image_train[idx])
	plt.title(f'Label : {label_train[idx]}')

plt.show()
"""
# build model
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(28,28)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation='tanh'))
model.add(tf.keras.layers.Dense(64,activation='sigmoid'))
model.add(tf.keras.layers.Dense(32,activation='relu'))
model.add(tf.keras.layers.Dense(16,activation='softmax'))


#compile model
model.compile(optimizer='RMSProp', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#review model

model.summary()

#train model

model.fit(image_train, label_train, epochs=6, batch_size=12) 
#evaluation
model.evaluate(image_test, label_test)

#save model 
model.save('mnist.h5')

