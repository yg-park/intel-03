# Homework03
Smart factory 불량 분류모델 training 결과

## Dataset 구조
```
(.otx)$ ds_count ./splitted_dataset 2
./splitted_dataset/:	126
./splitted_dataset/val:	26
./splitted_dataset/val/NOK:	11
./splitted_dataset/val/OK:	15
./splitted_dataset/train:	100
./splitted_dataset/train/NOK:	48
./splitted_dataset/train/OK:	52
```

## Training 결과
|Classification model|Accuracy|FPS|Training time|Batch size|Learning rate|Other hyper-prams|
|----|----|----|----|----|----|----|
|EfficientNet-V2-S| 1.0 | 41.684983 | 23.989454 | 64 | 0.0071 | |
|EfficientNet-B0| 1.0 | 54.278031 | 18.423660 | 64 | 0.0049 | |
|DeiT-Tiny| 1.0 | 60.941772 | 16.409106 | 64 | 0.0001 | |
|MobileNet-V3-large-1x| 1.0 | 77.017951 | 12.983986 | 64 | 0.0058 | |


## FPS 측정 방법
frame-time = 현재시간 - 이전시간 (수행시간) \
FPS = 1000 / frame-time 이므로, \
1000 / Training-time 공식으로 FPS를 측정했습니다.