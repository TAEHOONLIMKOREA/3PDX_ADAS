from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import uvicorn

# 현재 스크립트의 경로를 가져와서 디렉토리로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root ==> c:\Users\Taehoon\VSCodeProjects\3DP_3PDX_DIVA\backend
project_root = os.path.dirname(script_dir)
# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.append(project_root)

from Modules import TSAD_Preprocessor
from Modules.TSAD_Chronos import ChronosAnomalyDetector
from Modules.TSAD_Chronos_ver2 import add_anomaly_flags


# FastAPI 객체 생성
app = FastAPI()

# 루트 경로에 GET 요청 시 응답
@app.get("/")
def read_root():
    return {"message": "Hello FastAPI"}

@app.get("/test/inference_tsad")
def inference_tsad():
    # 1) 데이터 로드 (시간열이 없으면 자동으로 1분 간격 타임스탬프를 만듭니다)
    df = pd.read_csv("data/20220630_1035_Environment.csv")  # 예: cols = ["timestamp","cpu","mem","qps"]
    df = TSAD_Preprocessor.basic_preprocess(df)
    df = TSAD_Preprocessor.savgol_preprocess(df)
    # 2) 탐지기 생성 (최소 설정)
    detector = ChronosAnomalyDetector(
        context_length=1024,
        prediction_horizon=128,
        sliding_stride=32,
        robust_z_threshold=4.0,
        prefetch=True,  # 최초 1회 모델 스냅샷 다운로드
    )

    # 3) 이상치 탐지 (시간열이 "timestamp"라면 자동 인식됩니다)
    flagged_df = detector.detect(df)  # time_column 없으면 자동 탐색

    # 4) 결과 저장
    flagged_df.to_csv("metrics_with_anomaly.csv", index=False)


@app.get("/test/plot_anomaly")
def plot_anomaly():
    df = pd.read_csv("metrics_with_anomaly.csv")
    time_index = pd.to_datetime(df["Time"], errors="coerce", utc=False)

    for column_name in df.select_dtypes(include=[float, int]).columns:
        if column_name.endswith("_anomaly"):
            continue
        anomaly_col = f"{column_name}_anomaly"
        if anomaly_col not in df.columns:
            continue

        series_values = df[column_name].to_numpy()
        anomaly_flags = df[anomaly_col].astype(int).to_numpy() == 1
        anomaly_indices = np.where(anomaly_flags)[0]

        plt.figure(figsize=(10, 4))
        plt.plot(time_index, series_values, linewidth=1.0, label=column_name)
        if len(anomaly_indices):
            plt.scatter(time_index.iloc[anomaly_indices],
                        series_values[anomaly_indices],
                        s=28, marker="o", label="anomaly")
        plt.title(f"{column_name} (anomalies)")
        plt.xlabel("Time"); plt.ylabel(column_name)
        plt.grid(True, linestyle="--", alpha=0.3); plt.legend()
        plt.tight_layout()

        file_name = f"{column_name}_anomalies.png"
        plt.savefig(file_name, dpi=140)
        plt.close()
    


@app.post("/GetTimeseriesAD_CSV")
async def get_tsad_result(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    try:
        UPLOAD_DIR = Path("./temp_files")
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
    

# === 파일 맨 아래에 추가 ===
if __name__ == "__main__":

    # 환경변수로 기본값 제어 가능
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "55300"))
    reload_flag = os.getenv("RELOAD", "1") == "1"  # 개발 중엔 기본 True

    # reload를 쓰려면 "모듈경로:앱이름" 형태로 전달하는 게 가장 안정적
    module_name = Path(__file__).stem  # 예: main.py -> "main"
    uvicorn.run(
        f"{module_name}:app",
        host=host,
        port=port,
        reload=reload_flag,
        # 필요한 경우 로깅 레벨 등도 조정 가능:
        reload_dirs=["/home/taehoon/AD_API_Server/Server"],  # 프로젝트 소스만 감시
        reload_excludes=["venv/*", ".conda/*", "**/__pycache__/*"]
    )