# Face Blur System for Privacy Protection

This system was created for privacy protection. 

During real-time video recording, there is a frequent occurrence of privacy invasion when other individuals are inadvertently captured, compromising personal privacy. And manually blurring the faces of other people in the editing video process is not an easy task. So we developed a system using the OpenVINO AI model to automatically detect and recognize human faces, enabling the application of blur processing. In addition, we added features to automatically recognize specific individuals to avoid blurring, and the ability to adjust the intensity of the blur processing.

## Team Member
##### Yeajin Eum @Amaziniverse
##### Juhee Jeong @juhee67
##### Woosun Jin @abbblin 

---
## Team Up
<https://www.notion.so/Team-1-MiniProject-f4b3a9ff5d1f49599199adbfebe0cd01>

---
## Diagram
<img src="https://github.com/kccistc/intel-03/blob/MiniPrj-team1/class01/mini-project/team1/diagram.png" width=500 height=600/>

---

## How to run 

Download OpenVINO open model zoo. 
<https://github.com/openvinotoolkit/open_model_zoo>

```
# Set up the environment
source /opt/intel/openvino/bin/setupvars.sh

python3 face_recognition_demo.py \
  -i /dev/video0 \
  -m_fd intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml \
  -m_lm intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml \
  -m_reid intel/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml \
#  --verbose \
#  -fg 
``` 


## Result 
blur default 

![blur_default](https://github.com/kccistc/intel-03/blob/MiniPrj-team1/class01/mini-project/team1/blur_default.png)

blur weak

![blur_weak](https://github.com/kccistc/intel-03/blob/MiniPrj-team1/class01/mini-project/team1/blur_weak.png)

blur strong

![blur_strong](https://github.com/kccistc/intel-03/blob/MiniPrj-team1/class01/mini-project/team1/blur_strong.png)

