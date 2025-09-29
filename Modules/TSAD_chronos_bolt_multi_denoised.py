# chronos_anomaly_detector.py

import os
import re
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from huggingface_hub import snapshot_download
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame


# -----------------------------
# 데이터 구조
# -----------------------------
@dataclass
class ProcessedResult:
    processed_column: str
    base: str
    num_points: int
    anomaly_indices: List[int]
    num_anomalies: int


@dataclass
class OriginSummary:
    origin_column: str
    base: str
    num_points: int
    num_anomalies: int
    anomaly_ratio: float
    anomaly_indices: List[int]


# -----------------------------
# 유틸
# -----------------------------
TIME_COL_CANDIDATES = ["timestamp", "time", "date", "datetime"]


def find_time_index(df: pd.DataFrame, explicit_time_col: Optional[str] = None) -> pd.DatetimeIndex:
    """시간열 자동탐지(있으면 그 열 사용, 없으면 1분 간격 가짜 타임스탬프 생성)."""
    if explicit_time_col is not None and explicit_time_col in df.columns:
        ts = pd.to_datetime(df[explicit_time_col], errors="coerce", utc=False)
        if ts.notna().mean() > 0.9:
            return pd.DatetimeIndex(ts)
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
    # 1) leaderboard 기반
    try:
        lb = predictor.leaderboard(silent=True)
        if "score_val" in lb.columns and "model" in lb.columns:
            return lb.sort_values("score_val", ascending=True)["model"].iloc[0]
    except Exception:
        pass
    # 2) WeightedEnsemble 우선
    names = predictor.get_model_names()
    for n in names:
        if n.lower().startswith("weighteden"):
            return n
    # 3) Chronos 우선
    for n in names:
        if n.lower().startswith("chronos"):
            return n
    # 4) 첫 모델
    return names[0] if names else None


# -----------------------------
# 워커 프로세스 함수(피클링 가능한 인자만)
# -----------------------------
def _worker_run(
    gpu_id: int,
    cols_chunk: List[str],
    data_map: Dict[str, np.ndarray],
    time_index_values: np.ndarray,  # datetime64[ns] 배열
    context_len: int,
    horizon: int,
    stride: int,
    threshold_z: float,
    warmup_col: str,
    model_local_dir: str,
    hf_cache_dir: str,
) -> List[Dict[str, Any]]:
    """
    반환: ProcessedResult 사전(list)
    """
    # 환경세팅
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", hf_cache_dir)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    use_cuda = torch.cuda.is_available()
    device_str = "cuda" if use_cuda else "cpu"

    # Predictor 준비 (warmup)
    predictor = TimeSeriesPredictor(prediction_length=horizon, verbosity=0)

    warmup_series = data_map[warmup_col].astype(float)
    if np.isnan(warmup_series).any():
        warmup_series = pd.Series(warmup_series).interpolate(limit_direction="both").to_numpy()

    time_index = pd.DatetimeIndex(time_index_values)
    warmup_tsdf = to_tsdf(warmup_series, time_index, item_id=warmup_col)

    hparams = {
        "Chronos": {
            "model_path": model_local_dir,
            "device": device_str,
        },
    }

    predictor = predictor.fit(warmup_tsdf, hyperparameters=hparams)
    best_model = pick_best_model_name(predictor)

    # (A) 버퍼 준비
    residual_buf = {}
    for col in cols_chunk:
        s = data_map[col].astype(float)
        if np.isnan(s).any():
            s = pd.Series(s).interpolate(limit_direction="both").to_numpy()
        n = len(s)
        if n < context_len + horizon:
            # 빈 결과
            residual_buf[col] = {
                "series": s,
                "n": n,
                "sum": np.zeros(n, dtype=np.float32),
                "cnt": np.zeros(n, dtype=np.int32),
                "starts": [],
            }
            continue

        starts = list(range(context_len, n - horizon + 1, stride))
        residual_buf[col] = {
            "series": s,
            "n": n,
            "sum": np.zeros(n, dtype=np.float32),
            "cnt": np.zeros(n, dtype=np.int32),
            "starts": starts,
        }

    if not residual_buf:
        return []

    # (B) (col, st) 페어 → 메가배치
    MEGA_BATCH_ITEMS = 4096
    pairs = [(col, st) for col, buf in residual_buf.items() for st in buf["starts"]]

    for mb_start in tqdm(range(0, len(pairs), MEGA_BATCH_ITEMS),
                         desc=f"[GPU{gpu_id}] Mega-batches", leave=False, mininterval=0.5):
        chunk = pairs[mb_start: mb_start + MEGA_BATCH_ITEMS]

        frames = []
        for col, st in chunk:
            s = residual_buf[col]["series"]
            ctx_series = s[st - context_len: st]
            ctx_index  = time_index[st - context_len: st]
            frames.append(pd.DataFrame({
                "item_id": f"{col}__{st}",
                "timestamp": ctx_index,
                "target": ctx_series
            }))
        if not frames:
            continue

        batch_df  = pd.concat(frames, ignore_index=True)
        batch_tsdf = TimeSeriesDataFrame.from_data_frame(batch_df, id_column="item_id", timestamp_column="timestamp")

        preds = predictor.predict(batch_tsdf, model=best_model)
        pdf   = preds.to_data_frame()

        # item_id별 예측 ('0.5' or 'mean')
        yhat_map = {}
        for it, g in pdf.groupby(level=0):
            if "0.5" in g.columns:
                yhat_map[it] = g["0.5"].to_numpy()
            elif "mean" in g.columns:
                yhat_map[it] = g["mean"].to_numpy()
            else:
                yhat_map[it] = g.iloc[:, 0].to_numpy()

        # 잔차 누적
        for col, st in chunk:
            yhat = yhat_map[f"{col}__{st}"]
            s    = residual_buf[col]["series"]
            true_seg = s[st: st + horizon]
            res     = true_seg - yhat
            residual_buf[col]["sum"][st: st + horizon] += res
            residual_buf[col]["cnt"][st: st + horizon] += 1

    # (C) 결과 계산
    out_rows: List[Dict[str, Any]] = []
    for col, buf in residual_buf.items():
        n    = buf["n"]
        rs   = buf["sum"]; rc = buf["cnt"]
        mask = rc > 0

        residuals = np.zeros(n, dtype=np.float32)
        if np.any(mask):
            residuals[mask] = rs[mask] / rc[mask]

            med = np.median(residuals[mask])
            mad = np.median(np.abs(residuals[mask] - med))
            denom = mad if mad != 0 else 1e-9
            rzs = 0.6745 * (residuals - med) / denom
        else:
            rzs = np.zeros(n, dtype=np.float32)

        anoms = np.zeros(n, dtype=bool)
        if np.any(mask):
            anoms[mask] = np.abs(rzs[mask]) > threshold_z

        anomaly_idx_list = np.where(anoms)[0].astype(int).tolist()

        out_rows.append({
            "processed_column": col,
            "num_points": int(n),
            "anomaly_indices": anomaly_idx_list,
            "num_anomalies": int(len(anomaly_idx_list)),
        })

    return out_rows


# -----------------------------
# 메인 클래스
# -----------------------------
class ChronosAnomalyDetector:
    def __init__(
        self,
        context_len: int = 2048,
        horizon: int = 256,
        stride: int = 64,
        threshold_z: float = 4.0,
        model_id: str = "amazon/chronos-bolt-base",
        local_dir: str = "./models/chronos-bolt-base",
        hf_cache: str = "./hf_cache",
        prefetch: bool = True,
    ):
        """
        - context_len, horizon, stride: 예측 슬라이딩 윈도우 파라미터
        - threshold_z: robust z-score 임계값
        - model_id/local_dir/hf_cache: 모델 경로 설정
        - dtype: 'bfloat16' 또는 None
        - prefetch: 생성 시 모델 스냅샷 선다운로드 여부
        """
        self.context_len = context_len
        self.horizon = horizon
        self.stride = stride
        self.threshold_z = threshold_z
        self.model_id = model_id
        self.local_dir = os.path.abspath(local_dir)
        self.hf_cache = os.path.abspath(hf_cache)

        os.makedirs(self.hf_cache, exist_ok=True)
        os.makedirs(self.local_dir, exist_ok=True)

        if prefetch:
            self.prefetch_model()

    # -------------------------
    # 모델 사전 다운로드
    # -------------------------
    def prefetch_model(self):
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        os.environ.setdefault("HF_HOME", self.hf_cache)
        snapshot_download(
            repo_id=self.model_id,
            local_dir=self.local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
        )

    # -------------------------
    # 메인 진입점
    # -------------------------
    def detect(
        self,
        df: pd.DataFrame,
        time_col: Optional[str] = None,
        n_workers: Optional[int] = None,
        warmup_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        입력:
          - df: 전체 데이터프레임(원본+전처리 컬럼 포함)
          - time_col: 시간열 이름(없으면 자동탐지)
          - n_workers: 프로세스 수(기본: GPU 수 / 없으면 1)
          - warmup_col: 워밍업에 사용할 전처리 컬럼명(없으면 대상 중 첫 번째)

        반환:
          {
            "per_processed": List[ProcessedResult],
            "per_origin":    List[OriginSummary],
            "flags_df":      pd.DataFrame (timestamp + *_anomaly),
          }
        """
        # 시간 인덱스
        time_index = find_time_index(df, explicit_time_col=time_col)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in TIME_COL_CANDIDATES]
        target_cols = numeric_cols
            

        if len(target_cols) == 0:
            raise ValueError("전처리된 대상 컬럼이 없습니다.")

        # 워커 수
        num_gpus = torch.cuda.device_count()
        if n_workers is None:
            n_workers = num_gpus if num_gpus > 0 else 1
        if n_workers < 1:
            n_workers = 1

        # 데이터 매핑(필요 배열만)
        data_map = {c: df[c].to_numpy() for c in target_cols}

        # 워밍업 컬럼
        if warmup_col is None:
            warmup_col = target_cols[0]
        elif warmup_col not in target_cols:
            raise ValueError(f"warmup_col '{warmup_col}' 이(가) 대상 컬럼에 없습니다.")

        # 청크 분배
        chunks: List[List[str]] = [[] for _ in range(n_workers)]
        for i, col in enumerate(target_cols):
            chunks[i % n_workers].append(col)

        # 멀티프로세스 실행
        futures = []
        results_processed: List[ProcessedResult] = []
        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            for worker_id, cols_chunk in enumerate(chunks):
                if not cols_chunk:
                    continue
                futures.append(
                    ex.submit(
                        _worker_run,
                        worker_id,  # GPU id와 일치시킴
                        cols_chunk,
                        data_map,
                        time_index.values,
                        self.context_len,
                        self.horizon,
                        self.stride,
                        self.threshold_z,
                        warmup_col,
                        self.local_dir,
                        self.hf_cache,
                    )
                )

            for f in as_completed(futures):
                rows = f.result()
                for r in rows:
                    # base 이름 계산(워커에선 알 수 없으므로 여기서)
                    results_processed.append(
                        ProcessedResult(
                            processed_column=r["processed_column"],
                            base=base,
                            num_points=r["num_points"],
                            anomaly_indices=r["anomaly_indices"],
                            num_anomalies=r["num_anomalies"],
                        )
                    )

        # -------- (최종) 원본 단위 집계 --------
        base_to_indices: Dict[str, set] = {}
        for r in results_processed:
            idxs = set(r.anomaly_indices)
            base_to_indices.setdefault(r.base, set()).update(idxs)

        flags_df = pd.DataFrame({"timestamp": time_index})
        origin_summaries: List[OriginSummary] = []

        for base, idx_set in base_to_indices.items():
            if col not in df.columns:
                continue

            n = len(df[col])
            anom = np.zeros(n, dtype=bool)
            good_idxs = [i for i in idx_set if 0 <= i < n]
            if len(good_idxs):
                anom[np.array(good_idxs, dtype=int)] = True

            origin_summary = OriginSummary(
                origin_column=col,
                base=base,
                num_points=int(n),
                num_anomalies=int(anom.sum()),
                anomaly_ratio=float(anom.mean()),
                anomaly_indices=good_idxs,
            )
            origin_summaries.append(origin_summary)
            flags_df[f"{col}_anomaly"] = anom.astype(int)

        # 정렬: 이상치 많은 순
        origin_summaries.sort(key=lambda x: (-x.num_anomalies, x.origin_column))

        return {
            "per_processed": [asdict(r) for r in results_processed],
            "per_origin": [asdict(s) for s in origin_summaries],
            "flags_df": flags_df,
        }
