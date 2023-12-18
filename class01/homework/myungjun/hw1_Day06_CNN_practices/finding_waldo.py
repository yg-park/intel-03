# import libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import os

# tensorflow
import tensorflow as tf
#from tensorflow.python.keras.utils.vis_utils import plot_model
import pydot
from tensorflow.keras.utils import plot_model

# import time
import time

# get the images directory
image_dir = os.getcwd() + '/find_waldo/images/finding_waldo'

#print(image_dir)

# get the background and waldo image directory
background_dir = image_dir + '/wheres_wally.jpg'
waldo_dir = image_dir + '/waldo.png'
wilma_dir = image_dir + '/wilma.png'

# background image
background_im = Image.open(background_dir)
#background_im # jupyter notebook
#plt.imshow(background_im)
#plt.show()

# image of waldo
waldo_im = Image.open(waldo_dir)
waldo_im = waldo_im.resize((60, 100))
#plt.imshow(waldo_im)
#plt.show()

# image of wilma
wilma_im = Image.open(wilma_dir)
wilma_im = wilma_im.resize((60, 100))
#plt.imshow(wilma_im)
#plt.show()


# create a function to generate images
def generate_sample_image():
    #background image
    background_im = Image.open(background_dir)
    background_im = background_im.resize((500, 350))
    #background_im = Image.new("RGB", (500, 350), (255, 255, 255))

    # waldo
    waldo_im = Image.open(waldo_dir)
    waldo_im = waldo_im.resize((60, 100))

    # wilma
    wilma_im = Image.open(wilma_dir)
    wilma_im = wilma_im.resize((60, 100))

    # select x and y coordinates randomly we'll select between (0, 410) and (0, 250)
    col = np.random.randint(0, 410)
    row = np.random.randint(0, 230)

    # pick randomly between waldo and wilma. If 1 we will select waldo, if 0 we will select wilma
    rand_person = np.random.choice([0, 1], p = [0.5, 0.5])

    if rand_person == 1:
        background_im.paste(waldo_im, (col, row), mask = waldo_im)
        cat = 'Waldo'
    else:
        background_im.paste(wilma_im, (col, row), mask = wilma_im)
        cat = 'Wilma'
    
    return np.array(background_im).astype('uint8'), (col, row), rand_person, cat


# generate the sample image and plot
sample_im, pos, _, cat = generate_sample_image()
plt.imshow(sample_im)
plt.xticks([])
plt.yticks([])
plt.show()
