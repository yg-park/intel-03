# OpenVINO 패키지 설치
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov

import urllib.request
urllib.request.urlretrieve(
    url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
    filename='notebook_utils.py'
)

from notebook_utils import download_file

import ipywidgets as widgets

# openvino_notebooks 저장소에서 모델 다운로드
base_artifacts_dir = Path('./artifacts').expanduser()

model_name = "v3-small_224_1.0_float"
model_xml_name = f'{model_name}.xml'
model_bin_name = f'{model_name}.bin'

model_xml_path = base_artifacts_dir / model_xml_name

base_url = 'https://storage.openvinotoolkit.org/repositories/openvino_notebooks/models/mobelinet-v3-tf/FP32/'

if not model_xml_path.exists():
    download_file(base_url + model_xml_name, model_xml_name, base_artifacts_dir)
    download_file(base_url + model_bin_name, model_bin_name, base_artifacts_dir)
else:
    print(f'{model_name} already downloaded to {base_artifacts_dir}')
    
# OpenVINO Core 인스턴스 생성
core = ov.Core()

# 디바이스 선택 위젯 생성
device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

# 모델 컴파일
model = core.read_model(model=model_xml_path)
compiled_model = core.compile_model(model=model, device_name=device.value)

output_layer = compiled_model.output(0)

# 이미지 다운로드 및 전처리
image_filename = download_file(
    "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg",
    directory="data"
)
image = cv2.cvtColor(cv2.imread(str(image_filename)), code=cv2.COLOR_BGR2RGB)
input_image = cv2.resize(src=image, dsize=(224, 224))
input_image = np.expand_dims(input_image, 0)
plt.imshow(image)
plt.show()
# 추론 실행
result_infer = compiled_model([input_image])[output_layer]
result_index = np.argmax(result_infer)

# 클래스 정보 다운로드
imagenet_filename = download_file(
    "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/datasets/imagenet/imagenet_2012.txt",
    directory="data"
)
imagenet_classes = imagenet_filename.read_text().splitlines()
#imagenet_classes = ['background'] + imagenet_classes

# 결과 출력
print(imagenet_classes[result_index])
