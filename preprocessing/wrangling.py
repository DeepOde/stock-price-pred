import pandas as pd
import numpy as np
import os


def to_indi_csv( ticker, opfile, ohlcvfile=None, df=None):
    assert ohlcvfile is not None or df is not None, "Either OHLCV file or DataFrame must be provided."
    if df is None:
        df = pd.read_csv(ohlcvfile)
    df = df.dropna(subset=[ticker])
    t = df[['Date', 'Ticker', ticker]].pivot_table(index='Date', columns='Ticker')
    t.columns = t.columns.droplevel()
    t.reset_index(inplace=True)
    t.to_csv(opfile)

def get_labels(cls):
    move_dir = np.zeros_like(cls)
    cls_tmrw = cls.shift(-1)
    move_dir[cls_tmrw > cls] = 1
    move_dir[cls_tmrw <= cls] = -1
    move_dir[0] = np.nan
    move_dir[-1] = np.nan
    return move_dir, cls_tmrw
