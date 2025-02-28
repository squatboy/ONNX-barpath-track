import os
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from process_video import process_video_file
from config import TEMP_DIR

app = FastAPI()

# TEMP_DIR 폴더가 없으면 생성
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@app.post("/process_video/")
async def process_video_endpoint(file: UploadFile = File(...)):
    # 임시 파일 경로 생성 (temp/ 디렉토리 내에 저장)
    input_filename = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4().hex}.mp4")
    output_filename = os.path.join(TEMP_DIR, f"output_{uuid.uuid4().hex}.mp4")
    
    # 업로드된 동영상 저장
    with open(input_filename, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # 동영상 처리 실행 (객체 감지 및 이동 궤적 그리기)
    vertical_accuracy = process_video_file(input_filename, output_filename)
    
    # 필요에 따라 원시 입력 파일 삭제
    os.remove(input_filename)
    
    # 처리된 동영상 파일과 수직 정확도 반환
    return {
        "vertical_accuracy": round(vertical_accuracy, 2),
        "video_url": f"/get_video/{os.path.basename(output_filename)}"
    }

@app.get("/get_video/{filename}")
async def get_video(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(
            status_code=404,
            content={"message": "File not found"}
        )
    return FileResponse(file_path, media_type="video/mp4", filename="processed_video.mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)