import cv2
import numpy as np

def preprocess(image, img_size=640):
    """
    이미지 전처리 함수
      - 이미지를 img_size x img_size로 리사이즈
      - 픽셀 값 정규화, HWC → CHW로 차원 변경 및 배치 차원 추가
    """
    resized = cv2.resize(image, (img_size, img_size))
    img = resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img
