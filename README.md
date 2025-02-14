# onnx-barpathtrack-api

> Info

- Model : `YOLOv5s` to `ONNX`
- Datasets : COCO 데이터셋 대신 바벨을 인식하기 위한 커스텀 데이터셋을 구성 (단일 클래스로 학습진행)
- Performance Evaluation: `val.py`를 통해 `Precision`, `Recall`, `mAP` 등을 평가
- Epochs: 20
- 최종 mAP: 85%

Prob:
- 데이터셋 라벨링이 필요함
- 학습 데이터 수가 부족하여 과적합 발생 가능성 존재

해결 방안:
- `LabelImg`를 사용하여 직접 라벨링 수행
- 개선 가능성 : `Data augmentation` (좌우반전, 색상 변화 등) 적용하여 데이터 다양성 확보와 모델의 일반화 성능 향상이 가능

<br>

모델 학습 속도 저하와 GPU 메모리 부족 문제

- `batch size` 감소, `image size`를 640으로 제한하여 메모리 최적화
- `fp16` 옵션을 활성화하여 경량화 및 학습 속도 개선
```python
python train.py --img 640 --batch 16 --epochs 100 --data custom.yaml --weights yolov5s.pt
```

<br>

모델 기본 형태 `boxes`, `scores`, `labels` -> 클라이언트에 적용 시 후처리하여 JSON 형태로 변환이 필요
```python
def postprocess(output):
    boxes, scores, labels = output
    results = []
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:
            results.append({"bbox": box.tolist(), "score": float(score), "label": int(label)})
    return results
```
<br>

<img width="181" alt="스크린샷 2025-02-13 오전 11 28 54" src="https://github.com/user-attachments/assets/960004aa-ef30-482b-bfd9-2239457c67fc" />
