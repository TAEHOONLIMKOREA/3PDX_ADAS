from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import os
from datetime import datetime

# FastAPI 객체 생성
app = FastAPI()

# 루트 경로에 GET 요청 시 응답
@app.get("/")
def read_root():
    return {"message": "Hello FastAPI"}

# 파라미터 받기
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query": q}


@app.post("/upload")
async def upload_and_save_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    try:
        UPLOAD_DIR = Path("./uploads")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        orig_name = file.filename
        saved_name = f"{orig_name}.csv"
        
        saved_path = UPLOAD_DIR / saved_name

        with saved_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)

        # 원하면 응답 후 파일 삭제 (옵션)
        background_tasks.add_task(lambda p: os.remove(p) if os.path.exists(p) else None, str(saved_path))

        return FileResponse(
            path=str(saved_path),
            media_type="text/csv",
            filename=saved_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save/return error: {e}")