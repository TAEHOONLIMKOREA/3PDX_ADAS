import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def _savgol(series: pd.Series, window: int, poly: int) -> pd.Series:
    if window % 2 == 0: window += 1
    if window <= poly:   window = poly + 3 if poly % 2 == 0 else poly + 2
    s = series.copy()
    if s.isna().any(): s = s.interpolate(limit_direction="both")
    y = s.values.astype(float)
    try:
        yhat = savgol_filter(y, window_length=min(window, len(y) - (1 - len(y) % 2)), polyorder=poly, mode="interp")
    except Exception:
        yhat = y
    return pd.Series(yhat, index=series.index)

def savgol_preprocess(input_df: pd.DataFrame, window_size: int = 51, poly_order: int = 3) -> pd.DataFrame:
    """숫자형 컬럼에만 Savitzky–Golay 적용. timestamp 등 비숫자 컬럼은 그대로 유지."""
    out = input_df.copy()
    num_cols = [c for c in out.columns if np.issubdtype(out[c].dtype, np.number)]
    for c in num_cols:
        out[c] = _savgol(out[c], window_size, poly_order)
    return out

def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    time_cols = ["timestamp","time","date","datetime"]
    keep = [c for c in df.columns if c.lower() in time_cols]  # 타임스탬프 보존
    num_df = df.select_dtypes(include=[np.number])
    num_df = num_df.loc[:, ~(num_df.eq(0).all())]
    num_df = num_df.loc[:, ~num_df.apply(lambda s: set(np.unique(s.dropna())).issubset({0,1}))]
    num_df = num_df.loc[:, ~num_df.apply(lambda s: set(np.unique(s.dropna())).issubset({True,False}))]
    num_df = num_df.interpolate(limit_direction="both")
    return pd.concat([df[keep], num_df], axis=1)
