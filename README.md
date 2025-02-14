# onnx-barpathtrack-api

> Info

Model : Yolov5 to ONNX
Datasets : COCO 데이터셋 대신 바벨을 인식하기 위한 커스텀 데이터셋을 구성

Prob:
- 데이터셋 라벨링이 필요함
- 학습 데이터 수가 부족하여 과적합 발생 가능성 존재

해결 방안:
- LabelImg를 사용하여 직접 라벨링 수행
- Data augmentation 기법 (좌우반전, 색상 변화 등) 적용하여 데이터 다양성 확보

<br>

모델 학습 속도 저하와 GPU 메모리 부족 문제

- batch size 감소, image size를 640으로 제한하여 메모리 최적화
- fp16 옵션을 활성화하여 학습 속도 개선

```python
python train.py --img 640 --batch 16 --epochs 100 --data custom.yaml --weights yolov5s.pt
```

