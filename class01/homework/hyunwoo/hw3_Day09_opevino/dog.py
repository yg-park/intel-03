from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov

import urllib.request
urllib.request.urlretrieve(
    url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py",
    filename="notebook_utils.py",
)

from notebook_utils import download_file
    
import ipywidgets as widgets

import random
from typing import Optional

from typing import Dict

from openvino.runtime.utils.data_helpers import OVDict

model_dir = Path("model")
model_dir.mkdir(exist_ok=True)

# Create directory for TensorFlow model
tf_model_dir = model_dir / "tf"
tf_model_dir.mkdir(exist_ok=True)

# Create directory for OpenVINO IR model
ir_model_dir = model_dir / "ir"
ir_model_dir.mkdir(exist_ok=True)

model_name = "faster_rcnn_resnet50_v1_640x640"

openvino_ir_path = ir_model_dir / f"{model_name}.xml"

tf_model_url = "https://www.kaggle.com/models/tensorflow/faster-rcnn-resnet-v1/frameworks/tensorFlow2/variations/faster-rcnn-resnet50-v1-640x640/versions/1?tf-hub-format=compressed"

tf_model_archive_filename = f"{model_name}.tar.gz"

import tarfile

with tarfile.open(tf_model_dir / tf_model_archive_filename) as file:
    file.extractall(path=tf_model_dir)
    
download_file(
    url=tf_model_url,
    filename=tf_model_archive_filename,
    directory=tf_model_dir
)

ov_model = ov.convert_model(tf_model_dir)

# Save converted OpenVINO IR model to the corresponding directory
ov.save_model(ov_model, openvino_ir_path)

core = ov.Core()
device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)
openvino_ir_model = core.read_model(openvino_ir_path)
compiled_model = core.compile_model(model=openvino_ir_model, device_name=device.value)

model_inputs = compiled_model.inputs
model_input = compiled_model.input(0)
model_outputs = compiled_model.outputs

print("Model inputs count:", len(model_inputs))
print("Model input:", model_input)

print("Model outputs count:", len(model_outputs))
print("Model outputs:")
for output in model_outputs:
    print("  ", output)
    
image_path = Path("./data/coco.jpg")

download_file(
    url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
    filename=image_path.name,
    directory=image_path.parent,
)

# Read the image
image = cv2.imread(filename=str(image_path))

# The network expects images in RGB format
image = cv2.cvtColor(image, code=cv2.COLOR_BGR2RGB)

# Resize the image to the network input shape
resized_image = cv2.resize(src=image, dsize=(255, 255))

# Transpose the image to the network input shape
network_input_image = np.expand_dims(resized_image, 0)

inference_result = compiled_model(network_input_image)

_, detection_boxes, detection_classes, _, detection_scores, num_detections, _, _ = model_outputs

image_detection_boxes = inference_result[detection_boxes]
print("image_detection_boxes:", image_detection_boxes)

image_detection_classes = inference_result[detection_classes]
print("image_detection_classes:", image_detection_classes)

image_detection_scores = inference_result[detection_scores]
print("image_detection_scores:", image_detection_scores)

image_num_detections = inference_result[num_detections]
print("image_detections_num:", image_num_detections)

# Alternatively, inference result data can be extracted by model output name with `.get()` method
assert (inference_result[detection_boxes] == inference_result.get("detection_boxes")).all(), "extracted inference result data should be equal"

def add_detection_box(box: np.ndarray, image: np.ndarray, label: Optional[str] = None) -> np.ndarray:

    ymin, xmin, ymax, xmax = box
    point1, point2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
    box_color = [random.randint(0, 255) for _ in range(3)]
    line_thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1

    cv2.rectangle(img=image, pt1=point1, pt2=point2, color=box_color, thickness=line_thickness, lineType=cv2.LINE_AA)

    if label:
        font_thickness = max(line_thickness - 1, 1)
        font_face = 0
        font_scale = line_thickness / 3
        font_color = (255, 255, 255)
        text_size = cv2.getTextSize(text=label, fontFace=font_face, fontScale=font_scale, thickness=font_thickness)[0]
        # Calculate rectangle coordinates
        rectangle_point1 = point1
        rectangle_point2 = (point1[0] + text_size[0], point1[1] - text_size[1] - 3)
        # Add filled rectangle
        cv2.rectangle(img=image, pt1=rectangle_point1, pt2=rectangle_point2, color=box_color, thickness=-1, lineType=cv2.LINE_AA)
        # Calculate text position
        text_position = point1[0], point1[1] - 3
        # Add text with label to filled rectangle
        cv2.putText(img=image, text=label, org=text_position, fontFace=font_face, fontScale=font_scale, color=font_color, thickness=font_thickness, lineType=cv2.LINE_AA)
    return image

def visualize_dogs_only(inference_result: OVDict, image: np.ndarray, labels_map: Dict, detections_limit: Optional[int] = None):
    detection_boxes = inference_result.get("detection_boxes")
    detection_classes = inference_result.get("detection_classes")
    detection_scores = inference_result.get("detection_scores")
    num_detections = inference_result.get("num_detections")

    detections_limit = int(min(detections_limit, num_detections[0]) if detections_limit is not None else num_detections[0])

    # Normalize detection boxes coordinates to original image size
    original_image_height, original_image_width, _ = image.shape
    normalized_detection_boxes = detection_boxes[::] * [original_image_height, original_image_width, original_image_height, original_image_width]

    image_with_dogs_only = np.copy(image)

    for i in range(detections_limit):
        detected_class_id = int(detection_classes[0, i])
        score = detection_scores[0, i]

        # 개 클래스 ID가 17일 때 (객체 탐지 클래스 ID에 따라 변경)
        if detected_class_id == 18 and score > 0.4:  # 개 클래스이며, confidence가 일정 수준 이상인 경우에만 박스 표시
            add_detection_box(
                box=normalized_detection_boxes[0, i],
                image=image_with_dogs_only,
                label=f"Dog {score:.2f}",
            )

    plt.imshow(image_with_dogs_only)
    

coco_labels_file_path = Path("./data/coco_91cl.txt")

download_file(
    url="https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/data/dataset_classes/coco_91cl.txt",
    filename=coco_labels_file_path.name,
    directory=coco_labels_file_path.parent,
)

with open(coco_labels_file_path, "r") as file:
    coco_labels = file.read().strip().split("\n")
    coco_labels_map = dict(enumerate(coco_labels, 1))
    
#print(coco_labels_map)

visualize_dogs_only(
    inference_result=inference_result,
    image=image,
    labels_map=coco_labels_map,
    detections_limit=5,
)

plt.show()