#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Denoising & visualization utilities for sensor CSV time series.

- Moving Average
- Median Filter
- Savitzky–Golay

Usage:
    python TS_Preprocessor.py --csv ./data/preprocessing_with_anomaly.csv \
      --cols OxygenHighChamber TemperatureChamber PressureChamber  \
      --window 51 --poly 3 \
      --out ./data/preprocessing_with_anomaly_denoised.csv \
      --plotdir ./data/preprocessing_with_anomaly_denoised_plots
"""

from pathlib import Path

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def savgol(series: pd.Series, window: int, poly: int) -> pd.Series:
    if window % 2 == 0:
        window += 1
    if window <= poly:
        window = poly + 3 if poly % 2 == 0 else poly + 2
    s = series.copy()
    if s.isna().any():
        s = s.interpolate(limit_direction="both")
    y = s.values.astype(float)
    try:
        filtered = savgol_filter(y, window_length=window, polyorder=poly, mode="interp")
    except ValueError:
        filtered = y
    return pd.Series(filtered, index=series.index)


def apply_filters(df: pd.DataFrame, cols: list, window: int, poly: int) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = savgol(out[c], window, poly)
    return out

def savgol_preprocessor(input_df, wnidow_size: int | 51, poly_order: int | 3):

    cols = input_df.columns.tolist()
    for col in cols:
        if col not in input_df.columns:
            print(f"[warn] Column '{col}' not in CSV. Skipping.")
            continue
        if not np.issubdtype(input_df[col].dtype, np.number):
            print(f"[warn] Column '{col}' is not numeric. Skipping.")
            continue

    denoised_df = apply_filters(input_df, cols, wnidow_size, poly_order)
    
    return denoised_df
        

