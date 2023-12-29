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

### Team: [EOF] Separate Trash Collection
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
<점자를 인식하고 문장으로 만들어서 소리로 출력하는 프로젝트>
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
