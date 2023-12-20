## Required
```
python3 -m venv venv_openvino
source venv_openvino/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

# DYNAMIC DIARY mini-project

## Team members
1. 김명준
2. 박도현
3. 우이준

## Purpose
DYNAMIC DIARY는 사용자가 텍스트로 입력한 일기를 영상과 소리로 출력해주는 프로그램입니다. OpenVINO 프레임워크와 3가지 인공지능 모델을 사용하여 일기 내용을 소리내어 읽어주는 동시에 일기 내용과 연관된 영상(이미지)을 보여주고 연관된 음악을 들려줍니다.<br>
현재 DYNAMIC DIARY에서 일기 내용과 연관된 음악을 출력하는 모델과 코드는 음악을 저장하는데 문제가 발생하여 제외된 상태입니다.<br><br>
The DYNAMIC DIARY is a program which takes a text entered by the user and outputs the appropriate image and sounds. Three artificial intelligence models are used with the OpenVINO framework to read out loud the diary entry, while simutaneously displaying an image and music related to the contents of the diary entry.<br>
The accompanying music has been omitted, along with the corresponding A.I. model due to errors in saving the model output.<br><br>

## Diagram
![Text-to-Image](https://github.com/DohyeonP/Dynamic-Diary/assets/62331929/05edb978-165e-406f-b402-35c52fdbbda0)
Stable Diffusion (OpenVINO sample #225, a latent diffusion model)<br><br>

![Text-to-Audio](https://github.com/DohyeonP/Dynamic-Diary/assets/62331929/f5e40cc5-fa0d-4cd5-9853-a97c844a61b2)
MusicGen (OpenVINO sample #250, a single-stage autoregressive transformer model)<br><br>

![Text-to-Speech](https://github.com/DohyeonP/Dynamic-Diary/assets/62331929/cf554f3e-880e-4468-8b38-0839292d5ced)
Bark (OpenVINO sample #256, a multi-stage transformer model)<br><br>

## How to run
![Flow_diagram_details](https://github.com/DohyeonP/Dynamic-Diary/assets/62331929/40e9fc54-9c1f-43f1-a5b2-9f954cae7ad6)

## To use gemini
Get API key from "https://makersuite.google.com/app/apikey"

사용자가 일기를 작성하면 일기의 내용이 gemini 에게 전달되어서 일기에서 작성된 keyword 와 musical atmosphere를 얻는다.
해당 keyword를 기반으로 이미지가 만들어지고, musical atmosphere로 음악이 만들어지며 작성한 내용이 TTS로 변환되어 GUI에서 출력이 된다.



## Result
