import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
import pandas as pd
from itertools import product
import csv 
import os.path
import datetime


def sliding_window_cv_regression(X, y, pipe, n_tr, n_ts=1, scorers=[], comment="", post_processor=None):
    assert len(X) == len(y), "Length of X ([]) must match that of y ([]).".format(len(X), len(y))
    y_pred = []
    y_target = []
    agg_results = {}
    if post_processor is not None:
        post_processor_f, post_processor_args = post_processor[0], post_processor[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

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
        print(y_pred, y_target)
        
    agg_results['time'] = datetime.datetime.now()
    agg_results['model'] = str(pipe)
    agg_results['comment'] = comment
    for scorer in scorers:
        agg_results[scorer.__name__] = scorer(y_target, y_pred)
    
    return agg_results


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