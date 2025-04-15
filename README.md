# `ONNX`-barpathtrack-api
### Provides on-screen visualization of the barbell's movement and the accuracy of its vertical trajectory.

# Info

- **`Model`** : Trained **`YOLOv5s`** to **`ONNX`**
- **`Datasets`** : Custom dataset built for barbell recognition instead of COCO (trained with a single class)
- **`Performance Evaluation`**: Evaluated **`Precision`**, **`Recall`**, **`mAP`**, etc., using `val.py`
- **`Epochs`**: 20
- Final **`mAP`**: 86%

# Training 
![labels](https://github.com/user-attachments/assets/deb0684b-f103-4fd5-9920-bff4b9c33628)
![labels_correlogram](https://github.com/user-attachments/assets/a37e9b49-d5e2-4a15-9ef6-beaa87458b1b)

## API Endpoint Built
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)


## Logic
- Applied confidence-based filtering and Non-Maximum Suppression (NMS) -> selects representative detections
- Confidence threshold `thresh = 0.3`

<br>

## Examples
<img width="181" alt="Screenshot 2025-02-13 at 11 28 54 AM (KST)" src="https://github.com/user-attachments/assets/960004aa-ef30-482b-bfd9-2239457c67fc" />
<img width="302" alt="image" src="https://github.com/user-attachments/assets/06f8f567-aa53-4b9a-b686-9328df059b75" />


---


## Train Batches
![train_batch0](https://github.com/user-attachments/assets/1013e437-3f28-48a3-bdb3-e773770ddf05)

---

# Usage
## Requirements

```bash
pip install -r requirements.txt
```

## Running the Server

```bash
uvicorn main:app --reload
```

## API Endpoints

### `/process_video/` (POST)

Upload a video file for processing.

**Request Body (multipart/form-data):**

- `file`: Video file (.mp4).

**Example `curl`:**

```bash
curl -X POST -F "file=@your_video.mp4" [http://127.0.0.1:8000/process_video/](http://127.0.0.1:8000/process_video/)
```

**Response (JSON):**

```json
{
  "vertical_accuracy": 95.23,
  "video_url": "/get_video/output_unique_filename.mp4"
}
```

### `/get_video/{filename}` (GET)

Download the processed video.

**Example URL:**

`http://127.0.0.1:8000/get_video/output_unique_filename.mp4`

---

## `Prob`
- Dataset labeling is required.
  
- Potential for overfitting due to an insufficient number of training data.

## `Solution`
- Performed direct labeling using `LabelImg`.
  
- Potential improvements: Applying `Data augmentation` (horizontal flip, color variations, etc.) to secure data diversity and enhance the model's generalization performance.

<br>

## `Prob`
- Impact on trajectory and vertical accuracy in object unrecognition intervals.
(Connects the last recognized position and the newly recognized position with a straight line)

## `Solution`
- Handling of object unrecognition intervals and re-detection after prolonged object loss.
Setup variables related to interval re-detection:
```python
last_detection_frame = -1  # Frame number when the object was last detected
current_frame = 0  # Current frame number
max_frame_gap = int(fps * 1.5)  # Maximum allowed frame gap (1.5 seconds)
