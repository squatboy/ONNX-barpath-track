import cv2
import numpy as np
import onnxruntime as ort
from utils import preprocess
from config import MODEL_PATH

# 전역에서 ONNX 모델 로드 (재사용)
ort_session = ort.InferenceSession(MODEL_PATH)

def process_video_file(input_video_path: str, output_video_path: str):
    """
    업로드된 동영상 파일을 처리하여,
    - 프레임별로 객체 감지 (ONNX 모델 이용)
    - 빨간색 바운딩 박스로 객체 표시
    - 바운딩 박스 중심 좌표를 따라 형광초록색 얇은 선(두께 1)으로 이동 궤적 그리기
    후, 처리된 동영상을 output_video_path에 저장합니다.
    """
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    trajectory = []  # 이동 궤적 저장 (중심 좌표)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        img_h, img_w, _ = frame.shape
        input_image = preprocess(frame)
        
        # ONNX 모델 추론 실행
        ort_inputs = {ort_session.get_inputs()[0].name: input_image}
        ort_outs = ort_session.run(None, ort_inputs)
        
        raw_output = ort_outs[0][0]  # (25200, 6)
        # 모델 출력 순서: x_center, y_center, width, height, confidence, class
        x_center, y_center, width, height = raw_output[:, 0], raw_output[:, 1], raw_output[:, 2], raw_output[:, 3]
        confidences = raw_output[:, 4]
        class_ids = raw_output[:, 5]
        
        thresh = 0.3
        filtered_indices = np.where(confidences > thresh)
        if len(filtered_indices[0]) == 0:
            out.write(frame)
            continue
        
        filtered_x_center = x_center[filtered_indices]
        filtered_y_center = y_center[filtered_indices]
        filtered_width = width[filtered_indices]
        filtered_height = height[filtered_indices]
        filtered_confidences = confidences[filtered_indices]
        filtered_class_ids = class_ids[filtered_indices]
        
        # 좌표 변환: (x_center, y_center, w, h) → (x1, y1, x2, y2)
        x1 = filtered_x_center - (filtered_width / 2)
        y1 = filtered_y_center - (filtered_height / 2)
        x2 = filtered_x_center + (filtered_width / 2)
        y2 = filtered_y_center + (filtered_height / 2)
        
        # 전처리 시 640x640 사용 -> 원본 이미지 크기에 맞게 조정
        scale_x = img_w / 640
        scale_y = img_h / 640
        x1 *= scale_x
        y1 *= scale_y
        x2 *= scale_x
        y2 *= scale_y
        
        best_idx = np.argmax(filtered_confidences)
        best_box = (int(x1[best_idx]), int(y1[best_idx]), int(x2[best_idx]), int(y2[best_idx]))
        best_confidence = filtered_confidences[best_idx]
        best_class_id = int(filtered_class_ids[best_idx])
        
        # 빨간색 바운딩 박스 그리기
        cv2.rectangle(frame, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 0, 255), 2)
        cv2.putText(frame, f"{best_class_id}: {best_confidence:.2f}", (best_box[0], best_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 중심 좌표 계산 후 궤적 저장
        center_x = (best_box[0] + best_box[2]) // 2
        center_y = (best_box[1] + best_box[3]) // 2
        trajectory.append((center_x, center_y))
        
        # 형광초록색 얇은 선(두께 1)으로 이동 궤적 그리기
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
