from pathlib import Path

import tensorflow as tf

model_xml = Path("model/flower/flower_ir.xml")
dataset_url = (
    "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
)
data_dir = Path(tf.keras.utils.get_file("flower_photos", origin=dataset_url, untar=True))

if not model_xml.exists():
    print("Executing training notebook. This will take a while...")
    %run 301-tensorflow-training-openvino.ipynb


import sys

import matplotlib.pyplot as plt
import numpy as np
import nncf
from openvino.runtime import Core
from openvino.runtime import serialize
from PIL import Image
from sklearn.metrics import accuracy_score

sys.path.append("../utils")
from notebook_utils import download_file



img_height = 180
img_width = 180
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=1
)

for a, b in val_dataset:
    print(type(a), type(b))
    break



def transform_fn(data_item):
    """
    The transformation function transforms a data item into model input data.
    This function should be passed when the data item cannot be used as model's input.
    """
    images, _ = data_item
    return images.numpy()


calibration_dataset = nncf.Dataset(val_dataset, transform_fn)



core = Core()
ir_model = core.read_model(model_xml)


quantized_model = nncf.quantize(
    ir_model,
    calibration_dataset,
    subset_size=1000
)



compressed_model_dir = Path("model/optimized")
compressed_model_dir.mkdir(parents=True, exist_ok=True)
compressed_model_xml = compressed_model_dir / "flower_ir.xml"
serialize(quantized_model, str(compressed_model_xml))



import ipywidgets as widgets

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

device



def validate(model, validation_loader):
    """
    Evaluate model and compute accuracy metrics.

    :param model: Model to validate
    :param validation_loader: Validation dataset
    :returns: Accuracy scores
    """
    predictions = []
    references = []

    output = model.outputs[0]

    for images, target in validation_loader:
        pred = model(images.numpy())[output]

        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)

    scores = accuracy_score(references, predictions)

    return scores



original_compiled_model = core.compile_model(model=ir_model, device_name=device.value)
quantized_compiled_model = core.compile_model(model=quantized_model, device_name=device.value)

original_accuracy = validate(original_compiled_model, val_dataset)
quantized_accuracy = validate(quantized_compiled_model, val_dataset)

print(f"Accuracy of the original model: {original_accuracy:.3f}")
print(f"Accuracy of the quantized model: {quantized_accuracy:.3f}")



original_model_size = model_xml.with_suffix(".bin").stat().st_size / 1024
quantized_model_size = compressed_model_xml.with_suffix(".bin").stat().st_size / 1024

print(f"Original model size: {original_model_size:.2f} KB")
print(f"Quantized model size: {quantized_model_size:.2f} KB")



def pre_process_image(imagePath, img_height=180):
    # Model input format
    n, c, h, w = [1, 3, img_height, img_height]
    image = Image.open(imagePath)
    image = image.resize((h, w), resample=Image.BILINEAR)

    # Convert to array and change data layout from HWC to CHW
    image = np.array(image)

    input_image = image.reshape((n, h, w, c))

    return input_image




# Get the names of the input and output layer
# model_pot = ie.read_model(model="model/optimized/flower_ir.xml")
input_layer = quantized_compiled_model.input(0)
output_layer = quantized_compiled_model.output(0)

# Get the class names: a list of directory names in alphabetical order
class_names = sorted([item.name for item in Path(data_dir).iterdir() if item.is_dir()])

# Run inference on an input image...
inp_img_url = (
    "https://upload.wikimedia.org/wikipedia/commons/4/48/A_Close_Up_Photo_of_a_Dandelion.jpg"
)
directory = "output"
inp_file_name = "A_Close_Up_Photo_of_a_Dandelion.jpg"
file_path = Path(directory)/Path(inp_file_name)
# Download the image if it does not exist yet
if not Path(inp_file_name).exists():
    download_file(inp_img_url, inp_file_name, directory=directory)

# Pre-process the image and get it ready for inference.
input_image = pre_process_image(imagePath=file_path)
print(f'input image shape: {input_image.shape}')
print(f'input layer shape: {input_layer.shape}')

res = quantized_compiled_model([input_image])[output_layer]

score = tf.nn.softmax(res[0])

# Show the results
image = Image.open(file_path)
plt.imshow(image)
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
        class_names[np.argmax(score)], 100 * np.max(score)
    )
)




# print the available devices on this system
print("Device information:")
print(core.get_property("CPU", "FULL_DEVICE_NAME"))
if "GPU" in core.available_devices:
    print(core.get_property("GPU", "FULL_DEVICE_NAME"))




# Original model - CPU
! benchmark_app -m $model_xml -d CPU -t 15 -api async



# Quantized model - CPU
! benchmark_app -m $compressed_model_xml -d CPU -t 15 -api async



# Original model - MULTI:CPU,GPU
if "GPU" in core.available_devices:
    ! benchmark_app -m $model_xml -d MULTI:CPU,GPU -t 15 -api async
else:
    print("A supported integrated GPU is not available on this system.")



# Quantized model - MULTI:CPU,GPU
if "GPU" in core.available_devices:
    ! benchmark_app -m $compressed_model_xml -d MULTI:CPU,GPU -t 15 -api async
else:
    print("A supported integrated GPU is not available on this system.")



# print the available devices on this system
print("Device information:")
print(core.get_property("CPU", "FULL_DEVICE_NAME"))
if "GPU" in core.available_devices:
    print(core.get_property("GPU", "FULL_DEVICE_NAME"))



benchmark_output = %sx benchmark_app -m $model_xml -t 15 -api async
# Remove logging info from benchmark_app output and show only the results
benchmark_result = benchmark_output[-8:]
print("\n".join(benchmark_result))



benchmark_output = %sx benchmark_app -m $compressed_model_xml -t 15 -api async
# Remove logging info from benchmark_app output and show only the results
benchmark_result = benchmark_output[-8:]
print("\n".join(benchmark_result))



if "GPU" in core.available_devices:
    benchmark_output = %sx benchmark_app -m $model_xml -d MULTI:CPU,GPU -t 15 -api async
    # Remove logging info from benchmark_app output and show only the results
    benchmark_result = benchmark_output[-8:]
    print("\n".join(benchmark_result))
else:
    print("An GPU is not available on this system.")



if "GPU" in core.available_devices:
    benchmark_output = %sx benchmark_app -m $compressed_model_xml -d MULTI:CPU,GPU -t 15 -api async
    # Remove logging info from benchmark_app output and show only the results
    benchmark_result = benchmark_output[-8:]
    print("\n".join(benchmark_result))
else:
    print("An GPU is not available on this system.")


















