import numpy as np
from preprocessing.wrangling import get_indi_df, get_labels, slide_and_flatten
from preprocessing.extract_features import get_all_ta_features, get_wavelet_coeffs
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import pandas as pd
from itertools import product
import csv 
import os.path
import datetime


def add_closing_price(y, cls_price):
    return y + cls_price

def sliding_window_cv_regression(X, y, pipe, n_tr, n_ts=1, scorers=[], comment="", post_processor=None):
    assert len(X) == len(y), "Length of X ([]) must match that of y ([]).".format(len(X), len(y))
    y_pred = []
    y_target = []
    agg_results = {}
    if post_processor is not None:
        post_processor_f, post_processor_args = post_processor[0], post_processor[1]

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using {device} device")

    for i_tr_start in range(0, len(X)-(n_tr+n_ts)):
        # The last i_ts_end should be len(X).
        # i_ts_end = i_ts_start + n_ts
        # Now, i_tr_end = i_ts_start
        # So, i_tr_start = i_ts_start - n_tr
        # But, i_ts_start = i_ts_end - n_ts
        # Thus, i_tr_start = i_ts_end - n_tr - n_ts
        # Hence, last i_tr_start = len(X) - (n_tr + n_ts)

        i_tr_end = i_ts_start = i_tr_start + n_tr 
        i_ts_end = i_ts_start + n_ts 

        if isinstance(X, pd.DataFrame):
            Xtr, Xts = X.iloc[i_tr_start:i_tr_end, :], X.iloc[i_ts_start:i_ts_end, :]
        elif isinstance(X, np.ndarray):
            Xtr, Xts = X[i_tr_start:i_tr_end, :], X[i_ts_start:i_ts_end, :]
        ytr, yts = y[i_tr_start:i_tr_end], y[i_ts_start:i_ts_end]

        pipe.fit(Xtr, ytr)
        yts_hat = pipe.predict(Xts)
        y_pred.extend(yts_hat)
        y_target.extend(yts)
    
    if len(y_pred) > 1:
        y_pred = np.squeeze(y_pred)

    if post_processor is not None:
        y_pred = post_processor_f(y_pred, **post_processor_args)
        y_target = post_processor_f(y_target, **post_processor_args)
        # print(y_pred, y_target)
        
    agg_results['time'] = datetime.datetime.now()
    agg_results['model'] = str(pipe)
    agg_results['comment'] = comment
    for scorer in scorers:
        agg_results[scorer.__name__] = scorer(y_target, y_pred)
    
    return agg_results

def all_stock_estimator_test(list_dir, list_prefix, list_suffix, save_dir, save_prefix, save_suffix, results_file, estimator,
 n_tr, n_ts=1, scorers=[], len_window=10, keep_features=None, comment='', start_date="2017-01-01", end_date=None):
    results = []

    if keep_features is None:  # If features is None, use all features.
        keep_features = {'ohlcv', 'ta', 'wavelet', 'dct'}

    # i = 0
    for f in os.listdir(list_dir):
        if f.startswith(list_prefix) and f.endswith(list_suffix):
                savefile = os.path.join(save_dir, save_prefix+f[9:-8]+save_suffix)
                listfile = os.path.join(list_dir, f)
                p = pd.read_csv(listfile)
                symbols = list(p['Symbol'].values + '.NS')
                for symbol in symbols:
                    # if i == 2:
                    #     break
                    # i += 1
                    try:
                        end_date_str = end_date if end_date is not None else 'full'
                        comment = comment + symbol + '_' + start_date + '_' + end_date_str + '_' + 'wl' + str(len_window) + '_' + 'tr' + str(n_tr) + '_' + 'ts' + str(n_ts)
                        df = get_indi_df(symbol, ohlcvfile=savefile, start_date=start_date)
                        move_dir_target, cls_target = get_labels(df['Close'])
                        df = df.iloc[:-1]
                        cls_target = cls_target.iloc[:-1]
                        
                        drop_columns = ['Date', 'Adj Close']
                        if 'ta' in keep_features:
                            df = get_all_ta_features(df)
                        if 'ohlcv' not in keep_features:
                            drop_columns.extend(['Open', 'High', 'Low', 'Close', 'Volume'])
                        if 'wavelet' in keep_features:
                            df_wavelet = get_wavelet_coeffs(df['Close'], len_window=len_window, decomp_level=2)
                            df_wavelet = pd.DataFrame.from_records(df_wavelet, index=df.index)
                            df = df.merge(df_wavelet, left_index=True, right_index=True)

                        
                        unflattened_df_cls = df['Close']
                        df.drop(drop_columns, axis=1, inplace=True)

                        unflattened_df_index = df.index
                        df = slide_and_flatten(df, window_len=len_window)
                        df = pd.DataFrame(df, index=unflattened_df_index[(len_window-1):])
                        
                        # y = cls_target - df['Close']
                        y = cls_target[(len_window-1):] - unflattened_df_cls.iloc[(len_window-1):]
                        # y30 = cls_target[29:] - df['Close'].iloc[29:]
                        # y60 = cls_target[59:] - df['Close'].iloc[59:]
                        
                        # df30 = slide_and_flatten(df, window_len=30)
                        # df30 = pd.DataFrame(df30, index=df.index[29:])
                        # df60 = slide_and_flatten(df, window_len=60)
                        # df60 = pd.DataFrame(df60, index=df.index[59:])

                        # df10_wavelet = get_wavelet_coeffs(df['Close'], len_window=10, decomp_level=2)
                        # df10_wavelet = pd.DataFrame.from_records(df10_wavelet, index=df10.index)
                        # df30_wavelet = get_wavelet_coeffs(df['Close'], len_window=30, decomp_level=2)
                        # df30_wavelet = pd.DataFrame.from_records(df30_wavelet, index=df30.index)
                        # df60_wavelet = get_wavelet_coeffs(df['Close'], len_window=60, decomp_level=2)
                        # df60_wavelet = pd.DataFrame.from_records(df60_wavelet, index=df60.index)

                        # df = df.merge(df_wavelet, left_index=True, right_index=True)
                        # df30 = df30.merge(df30_wavelet, left_index=True, right_index=True)
                        # df60 = df60.merge(df60_wavelet, left_index=True, right_index=True)

                        post_processor = (add_closing_price, {'cls_price':unflattened_df_cls.iloc[(len_window-1):len(unflattened_df_cls)-(n_tr+n_ts)]})
                        result = sliding_window_cv_regression(df, y, estimator, n_tr, n_ts, scorers, comment=comment, post_processor=post_processor)
                        results.append(result)
                        print("Result obtained for {}".format(symbol))
                    except Exception as e:
                        print("An error occured while testing on a stock {}".format(symbol))
                        print(e)
                        print("Skipped.")

    if results_file is not None:
        file_exists = os.path.isfile(results_file)
    
    with open(results_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter=',', lineterminator='\n')

        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        writer.writerows(results)

    print("Completed all stocks test.")                    




def batch_test_swcv_regression(list_X, list_y, list_pipe, list_n_tr, list_n_ts, scorers, list_post_processors=None, savefile=None, comment_X=None, comment_y=None):
    results = []
    for i in range(len(list_X)):
        X, y = list_X[i], list_y[i]
        if list_post_processors:
            post_processor = list_post_processors[i]
        else:
            post_processor = None
        comment = []
        if comment_X is not None:
            comment.append(comment_X[i])
        if comment_y is not None:
            comment.append(comment_y[i])
        for pipe, n_tr, n_ts in product(list_pipe, list_n_tr, list_n_ts):
            comment.append(n_tr)
            comment.append(n_ts)
            result = sliding_window_cv_regression(X, y, pipe, n_tr, n_ts, scorers, "_".join([str(c) for c in comment]), post_processor)
            results.append(result)
            print("A test completed. (Comment : {})".format(comment))
            comment.pop()
            comment.pop()
        
    if savefile is not None:
        file_exists = os.path.isfile(savefile)
        
        with open(savefile, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter=',', lineterminator='\n')

            if not file_exists:
                writer.writeheader()  # file doesn't exist yet, write a header

            writer.writerows(results)