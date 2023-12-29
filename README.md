# 상공회의소 서울기술교육센터 인텔교육 3기

## Clone code 

```shell
git clone --recurse-submodules https://github.com/kccistc/intel-03.git
```

* `--recurse-submodules` option 없이 clone 한 경우, 아래를 통해 submodule update

```shell
git submodule update --init --recursive
```

## Preparation

### Git LFS(Large File System)

* 크기가 큰 바이너리 파일들은 LFS로 관리됩니다.

* git-lfs 설치 전

```shell
# Note bin size is 132 bytes before LFS pull

$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 132 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

* git-lfs 설치 후, 다음의 명령어로 전체를 가져 올 수 있습니다.

```shell
$ sudo apt install git-lfs

$ git lfs pull
$ find ./ -iname *.bin|xargs ls -l
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 3358630 Nov  6 09:41 ./mosaic-9.bin
-rw-rw-r-- 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
-rwxrwxr-x 1 <ID> <GROUP> 8955146 Nov  6 09:41 ./ssdlite_mobilenet_v2_fp16.bin
```

### 환경설정

* [Ubuntu](./doc/environment/ubuntu.md)
* [OpenVINO](./doc/environment/openvino.md)
* [OTX](./doc/environment/otx.md)

## Team project

### Team: EOF - Separate Trash Collection
최상의 재활용 원료 품질을 확보하기 위해, 분류된 재활용 쓰레기 카테고리를 한번 더 분리해준다. <br> 
근무자는 음성 명령을 통해 재활용 쓰레기 카테고리에 맞는 재분류 모델을 스위칭 해가며 작업을 진행한다. 
* Members
  | Name | Role |
  |----|----|
  | 권강현 | Project lead, 프로젝트를 총괄하고 망하면 책임진다. <br> Project manager, 마일스톤을 생성하고 프로젝트 이슈 진행상황을 관리한다. |
  | 박도현 | Architect, 프로젝트의 component를 구성하고 상위 디자인을 책임진다. |
  | 박용근 | AI modeling, 원하는 결과가 나오도록 AI model을 선택, data 수집, training을 수행한다. |
  | 우창민 | UI design, 사용자 인터페이스를 정의하고 구현한다. |
* Project Github : https://github.com/yg-park/EOF_SeparateTrashCollection.git
* 발표자료 : https://github.com/yg-park/EOF_SeparateTrashCollection/blob/main/Documents/EOF_presentation.pptx


### Team: 점자의 소리
점자를 인식하고 문장으로 만들어서 소리로 출력하는 프로젝트
* Members
  | Name | Role |
  |----|----|
  | 홍종현 | Project lead |
  | 강재환 | Project manager |
  | 김진완 | UI design |
  | 김혁구 | AI modeling |
  | 박진욱 | Architect |
* Project Github : https://github.com/myreporthjh/SoB.git
* 발표자료 : 


### Team: BTS(Balsa Tracking System)
표적에 따른 격추 자동화 시스템
* Members
  | Name | Role |
  |----|----|
  | 김규원 | Project lead, 프로젝트를 총괄하고 망하면 책임진다. |
  | 김지민 | Project manager, 마일스톤을 생성하고 프로젝트 이슈 진행상황을 관리한다. |
  | 왕정현 | AI modeling, 원하는 결과가 나오도록 AI model을 선택, data 수집, training을 수행한다. |
  | 이현우 | Architect, 프로젝트의 component를 구성하고 상위 디자인을 책임진다. |
  | 정주환 | HW manager , 전반적인 하드웨어 구성과 제작을 총괄한다.|
* Project Github : https://github.com/oz971124/BTS.git 
* 발표자료 : https://github.com/oz971124/BTS/blob/main/presentation.pptx


### Team: 누가바 - Face Privacy(손 모션 인식을 통한 사생활 보호 솔루션)

최근, 많은 사람들이 가까운 지인과의 소식 전달을 위하여 Social Network Service(이하 'SNS'라 지칭함)를 이용하고 있다. 그러나, 사용자가 사>진이나 영상을 촬영하여 SNS 서비스 플랫폼에 업로드하여 게시하고자 할 경우, 무분별한 타인의 얼굴 등 의도하지 않은 정보가 유출될 수 있어 문
제가 발생되기도 한다. <br>
이에 일부 사용자들의 경우 SNS 서비스 플랫폼에 사진을 업로드하고자 할 경우, 개인 정보가 유출될 수 있는 특정 영역을 모자이크 처리한 후 업>로드하고 있다. 예컨대, 관련성이 없는 타인의 얼굴, 차량 번호판 등을 직접 모자이크 처리한 후 업로드하는 방식이다. <br>
그러나, 종래의 방식에서는 사용자가 직접 사진 하나하나 특정 영역을 지정하여 모자이크 처리하는 방식으로, 업로드에 많은 시간이 소요되며, 사
용자의 불편함이 야기된다는 문제점이 있다. <br>
본 발명은 상기한 종래의 문제점을 해결하기 위해 제안된 것으로서, 영상에서 특정 영역만을 모자이크 처리하여 정보 게시 서비스를 통해 게시하>고, 개인정보 공개를 원하는 사용자에 따라 특정 손 모션 동작을 인식시켜 선별적으로 모자이크를 해제하여 배포할 수 있는 영상 모자이크 처리 >방법 및 이를 위한 장치를 제공한다. <br>
특히, 본 발명은 영상에서 얼굴 영역을 추출하고, 추출된 얼굴 영역에 대응하여 얼굴 키 값을 이용하여 해당 얼굴 영역을 모자이크 처리하고 영상
을 중계 및 재생하며, 영상 배포 시, 출연자의 개인정보 공개 여부에 따라 손 모션 동작에 의해 선별적으로 모자이크를 해제한 후 중계 및 배포할
 수 있는 영상 모자이크 처리 방법 및 이를 위한 장치를 제공하는 데 그 목적이 있다.<br><br>


* Members
  | Name | Role |
  |----|----|
  | 김명준 | Project lead, 프로젝트를 총괄한다. |
  | 엄예진 | Gesture detection model, 손동작을 인식하는 모델을 만든다. |
  | 우이준 | Dataset and documentation, 모델 훈련용 데이터 제작 및 문서화의 주 담당자이다. |
  | 정주희 | Gesture detection model, 손동작을 인식하는 모델을 만든다. |
  | 진우선 | Emotion detection model, 인물의 감정을 인식하는 모델을 만든다. |
* Project Github : https://github.com/Team-Intel-Edge-AI/OnTheEdge.git
* 발표자료 : https://github.com/Team-Intel-Edge-AI/OnTheEdge/blob/main/doc/Face_Privacy.pptx
