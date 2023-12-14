# import libs
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time



# load dataset
(image_train, label_train),(image_test, label_test) = tf.keras.datasets.mnist.load_data()



# read model
read_model = tf.keras.saving.load_model("mnist.h5")



# which test dataset to check
index = 33



# prepare image
image = np.array([image_test[index]])



# measure time
start = time.time()
predicted_value = read_model(image)
print("inference time : ", (time.time() - start)*1000, "msec")



# print label value
print("Label : ", label_test[index])
print("Result : ", np.argmax(predicted_value))



# print output
print(predicted_value)
print()



# inference & draw
