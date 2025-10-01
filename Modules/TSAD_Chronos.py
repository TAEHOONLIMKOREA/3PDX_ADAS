# chronos_anomaly_detector.py

import os
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from huggingface_hub import snapshot_download
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from chronos import BaseChronosPipeline  # <- inference 전용

TIME_COL_CANDIDATES = ["timestamp", "time", "date", "datetime"]
HF_CACHE_DIR = "./AD_API_Server/hf_cache"
MODEL_ID = "amazon/chronos-bolt-base"
MODEL_DIR = "./AD_API_Server/models/chronos-bolt-base"
MEGA_BATCH_ITEMS = 4096
# -----------------------------
# 유틸
# -----------------------------
def find_time_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """시간열 자동탐지(있으면 그 열 사용, 없으면 1분 간격 가짜 타임스탬프 생성)."""
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            ts = pd.to_datetime(df[c], errors="coerce", utc=False)
            if ts.notna().mean() > 0.9:
                return pd.DatetimeIndex(ts)
    return pd.date_range("2000-01-01", periods=len(df), freq="T")


def to_tsdf(series: np.ndarray, idx: pd.DatetimeIndex, item_id: str) -> TimeSeriesDataFrame:
    df = pd.DataFrame({"item_id": item_id, "timestamp": idx, "target": series})
    return TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")


def pick_best_model_name(predictor: TimeSeriesPredictor) -> Optional[str]:
    try:
        lb = predictor.leaderboard(silent=True)
        columns = lb.columns
        if "score_val" in columns and "model" in columns:
            models = lb.sort_values("score_val", ascending=True)["model"]
            model = models.iloc[0]
            return model
    except Exception as e:
        print(e)
        return None


# -----------------------------
# 워커 프로세스 함수(피클링 가능한 인자만)
# -----------------------------
def _worker_run(
    gpu_id: int,
    cols_chunk: List[str],
    input_df: pd.DataFrame,
    time_index: pd.DatetimeIndex,
    context_len: int,
    horizon: int,
    sliding_stride: int,
    robust_z_threshold: float,
    warmup_col: str,
) -> List[Dict[str, Any]]:
    
    # 환경세팅
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # Predictor 준비 (warmup)
    predictor = TimeSeriesPredictor(prediction_length=horizon, verbosity=0)

    warmup_series = input_df[warmup_col].astype(float).to_numpy()
    warmup_series = pd.Series(warmup_series).interpolate(limit_direction="both").to_numpy()
    warmup_tsdf = to_tsdf(warmup_series, time_index, item_id=warmup_col) 
    
    predictor = predictor.fit(
        warmup_tsdf,
        hyperparameters={
            "Chronos": {
                "model_path": MODEL_DIR,
                "device": "cuda",
                "dtype": "bfloat16",   # 5090 권장. 문제 시 주석 처리
            },
        },
    )
    best_model = pick_best_model_name(predictor)

    # (A) 버퍼 준비
    residual_buf = {}
    for col_name in cols_chunk:
        numeric_series = input_df[col_name].astype(float).to_numpy()
        if np.isnan(numeric_series).any():
            numeric_series = pd.Series(numeric_series).interpolate(limit_direction="both").to_numpy()

        series_len = len(numeric_series)
        if series_len < context_len + horizon:
            residual_buf[col_name] = {
                "series": numeric_series,
                "series_len": series_len,
                "residuals_sum": np.zeros(series_len, dtype=np.float32),
                "residuals_cnt": np.zeros(series_len, dtype=np.int32),
                "window_starts": [],
            }
            continue

        window_starts = list(range(context_len, series_len - horizon + 1, sliding_stride))
        residual_buf[col_name] = {
            "series": numeric_series,
            "series_len": series_len,
            "residuals_sum": np.zeros(series_len, dtype=np.float32),
            "residuals_cnt": np.zeros(series_len, dtype=np.int32),
            "window_starts": window_starts,
        }

    if not residual_buf:
        return []

    # (B) (col_name, start_index) 페어 → 메가배치
    column_start_pairs = []    
    for col, buf in residual_buf.items():
        for start_index in buf["window_starts"]:
            column_start_pairs.append((col, start_index))    

    for batch_start_index in tqdm(
        range(0, len(column_start_pairs), MEGA_BATCH_ITEMS),
        desc=f"[GPU{gpu_id}] Mega-batches",
        leave=False,
        mininterval=0.5,
    ):
        pair_chunk = column_start_pairs[batch_start_index: batch_start_index + MEGA_BATCH_ITEMS]

        mini_frames = []
        for col_name, start_index in pair_chunk:
            series_values = residual_buf[col_name]["series"]
            context_values = series_values[start_index - context_len: start_index]
            context_index = time_index[start_index - context_len: start_index]
            mini_frames.append(pd.DataFrame({
                "item_id": f"{col_name}__{start_index}",
                "timestamp": context_index,
                "target": context_values
            }))

        if not mini_frames:
            continue

        batch_dataframe = pd.concat(mini_frames, ignore_index=True)
        batch_tsdf = TimeSeriesDataFrame.from_data_frame(batch_dataframe, id_column="item_id", timestamp_column="timestamp")

        pred_frame = predictor.predict(batch_tsdf, model=best_model)
        pred_df = pred_frame.to_data_frame()

        # item_id별 예측 ('0.5' or 'mean')
        yhat_map = {}
        for item_id, pred_group in pred_df.groupby(level=0):
            if "0.5" in pred_group.columns:
                yhat_map[item_id] = pred_group["0.5"].to_numpy()
            elif "mean" in pred_group.columns:
                yhat_map[item_id] = pred_group["mean"].to_numpy()
            else:
                yhat_map[item_id] = pred_group.iloc[:, 0].to_numpy()

        # 잔차 누적
        for col_name, start_index in pair_chunk:
            forecast_values = yhat_map[f"{col_name}__{start_index}"]
            true_series = residual_buf[col_name]["series"]
            true_segment = true_series[start_index: start_index + horizon]
            residual_segment = true_segment - forecast_values

            residual_buf[col_name]["residuals_sum"][start_index: start_index + horizon] += residual_segment
            residual_buf[col_name]["residuals_cnt"][start_index: start_index + horizon] += 1

    # (C) 결과 계산 (컬럼별 anomaly index)
    output_rows = []
    for col_name, buf in residual_buf.items():
        total_length = buf["series_len"]
        residuals_sum = buf["residuals_sum"]
        residuals_cnt = buf["residuals_cnt"]
        pred_mask = residuals_cnt > 0

        averaged_residuals = np.zeros(total_length, dtype=np.float32)
        averaged_residuals[pred_mask] = residuals_sum[pred_mask] / residuals_cnt[pred_mask]
        if np.any(pred_mask):
            median_value = np.median(averaged_residuals[pred_mask])
            median_absolute_deviation = np.median(np.abs(averaged_residuals[pred_mask] - median_value))
            mad_denom = median_absolute_deviation if median_absolute_deviation != 0 else 1e-9
            robust_z_scores = 0.6745 * (averaged_residuals - median_value) / mad_denom
        else:
            robust_z_scores = np.zeros(total_length, dtype=np.float32)

        anomaly_boolean = np.zeros(total_length, dtype=bool)
        if np.any(pred_mask):
            anomaly_boolean[pred_mask] = np.abs(robust_z_scores[pred_mask]) > robust_z_threshold

        anomaly_index_list = np.where(anomaly_boolean)[0].astype(int).tolist()

        output_rows.append({
            "column": col_name,
            "anomaly_indices": anomaly_index_list,
            "num_points": int(total_length),
        })

    return output_rows


# -----------------------------
# 메인 클래스
# -----------------------------
class ChronosAnomalyDetector:
    def __init__(
        self,
        context_len: int = 1024,
        pred_horizon: int = 128,
        sliding_stride: int = 32,
        robust_z_threshold: float = 4.0,
        prefetch: bool = True,
    ):
        """
        - context_len, horizon, sliding_stride: 예측 슬라이딩 윈도우 파라미터
        - robust_z_threshold: robust z-score 임계값
        - model_id/local_model_dir/hf_cache_dir: 모델 경로 설정
        - prefetch: 생성 시 모델 스냅샷 선다운로드 여부
        """
        self.context_len = context_len
        self.horizon = pred_horizon
        self.sliding_stride = sliding_stride
        self.robust_z_threshold = robust_z_threshold

        os.makedirs(HF_CACHE_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

        if prefetch:
            self.prefetch_model()

    # -------------------------
    # 모델 사전 다운로드
    # -------------------------
    def prefetch_model(self):
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
        )

    # -------------------------
    # 메인 진입점
    # -------------------------
    def detect(
        self,
        input_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        입력:
          - dataframe: 전체 데이터프레임(원본)
          - time_column: 시간열 이름(없으면 자동탐지)
          - num_workers: 프로세스 수(기본: GPU 수 / 없으면 1)
          - warmup_column: 워밍업에 사용할 컬럼명(없으면 대상 중 첫 번째)

        반환:
          - 원본 DataFrame에 각 수치 컬럼마다 `<컬럼명>_anomaly` 열을 추가한 DataFrame
        """
        # 시간 인덱스
        time_index = find_time_index(input_df)

        # 대상 컬럼: 숫자형만
        numeric_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in TIME_COL_CANDIDATES]

        if len(numeric_cols) == 0:
            raise ValueError("이상치 탐지 대상이 될 숫자 컬럼이 없습니다.")

        # 워커 수
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            num_gpus = 1

        # 워밍업 컬럼
        warmup_col = numeric_cols[0]            

        # 청크 분배
        chunks: list[list[str]] = [[] for _ in range(num_gpus)]
        for i, col in enumerate(numeric_cols):
            chunks[i % num_gpus].append(col)

        # 멀티프로세스 실행
        futures = []
        aggregated_rows = []
        context = mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=num_gpus, mp_context=context) as executor:
            for worker_id, cols_chunk in enumerate(chunks):
                if not cols_chunk:
                    continue
                futures.append(
                    executor.submit(
                        _worker_run,
                        worker_id,  # GPU id와 일치시킴
                        cols_chunk,
                        input_df,
                        time_index,
                        self.context_len,
                        self.horizon,
                        self.sliding_stride,
                        self.robust_z_threshold,
                        warmup_col,
                    )
                )

            for future in as_completed(futures):
                worker_rows = future.result()
                aggregated_rows.extend(worker_rows)

        # 결과를 원본 DataFrame에 머지
        output_df = input_df.copy()
        for row_dict in aggregated_rows:
            target_col = row_dict["column"]
            target_len = len(output_df[target_col])
            anomalies = np.zeros(target_len, dtype=bool)

            valid_indices = [int(idx) for idx in row_dict["anomaly_indices"] if 0 <= int(idx) < target_len]
            if len(valid_indices) > 0:
                anomalies[np.array(valid_indices, dtype=int)] = True

            output_df[f"{target_col}_anomaly"] = anomalies.astype(int)

        return output_df
