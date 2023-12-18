# import
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# dataset load
fashion_mnist = tf.keras.datasets.fashion_mnist

(image_train, label_train), (image_test, label_test) = fashion_mnist.load_data()
print("Train Image Shape : ", image_train.shape)
print("Train Label : ", label_train.shape)
print("Test Image Shape : ", image_test.shape)
print("Test Label : ", label_test.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


### show
NUM = 20

plt.figure(figsize = (15, 15))
plt.subplots_adjust(hspace = 1)
for idx in range(NUM) : 
    sp = plt.subplot(5, 5, idx+1)
    plt.imshow(image_train[idx])
    plt.title(f"{class_names[label_train[idx]]}")

plt.show()



## 간단한 이미지 전처리 for ANN, from professor
image_train = image_train / 255.0
image_test = image_test / 255.0



# build model
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation = "relu"),
    keras.layers.Dense(10, activation = "softmax"),
])

'''
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape = (28, 28)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation = 'tanh'))
model.add(tf.keras.layers.Dense(64, activation = 'sigmoid'))
model.add(tf.keras.layers.Dense(32, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))
'''


# review model
model.summary()



# compile moddel
model.compile(optimizer = "Adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])



# train model
model.fit(image_train, label_train, epochs = 20) # batch_size = 12



# evaluate model
predictions = model.predict(image_test)

predictions[0]

np.argmax(predictions[0])

label_test[0]

def plot_image(i, predictions_array, true_label, img) : 
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label : 
        color = "blue"
    else : 
        color = "red"

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]), 
                                color = color)


def plot_value_array(i, predictions_array, true_label) : 
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize = (2*2*num_cols, 2*num_rows))

for i in range(num_images) : 
    plt.subplot(num_rows, 2*num_cols, 2*i + 1)
    plot_image(i, predictions, label_test, image_test)
    plt.subplot(num_rows, 2*num_cols, 2*i + 2)
    plot_value_array(i, predictions, label_test)

plt.show()


from sklearn.metrics import accuracy_score
print("accuracy score : ", accuracy_score(tf.math.argmax(predictions, -1), label_test))



'''
model.evaluate(image_test, label_test)


# save model 
model.save("fashion_mnist.h5")
'''





