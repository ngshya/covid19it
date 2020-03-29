from numpy import concatenate, flip, array, max, min, argmin, abs, isnan, array, vstack
from scipy.stats import pearsonr
from multiprocessing import Pool
from functools import partial
from .logger import logger


def compute_diff(v1, v2):
    v1, v2 = array(v1), array(v2)
    #v1, v2 = (v1-min(v1)) / max(v1), (v2-min(v2)) / max(v2)
    return pearsonr(v1, v2)[0]


def get_corr_lag(v1, v2, min_corr=0.9, max_lag=7):
    
    v1, v2 = array(v1), array(v2)
    #v1, v2 = (v1-min(v1)) / max(v1), (v2-min(v2)) / max(v2)
    l1, l2 = len(v1), len(v2)

    assert l1 == l2, "Input arrays with different lengths."

    max_lag = min((max_lag, l1-2))

    out_backward = flip([compute_diff(v1[j:], v2[:l2-j]) for j in range(max_lag+1)])
    out_forward = [compute_diff(v1[:l1-j], v2[j:]) for j in range(1, max_lag+1)]
    out = concatenate((out_backward, out_forward), axis=0)
    out[isnan(out)] = -2
    array_idx = array(range(-max_lag, max_lag+1))
    value_max_corr = max(out)
    if value_max_corr >= min_corr:
        array_idx_max_corr = array_idx[abs(out - value_max_corr) < 1e-9]
        array_abs_idx_max_corr = abs(array_idx_max_corr)
        lag = array_idx_max_corr[argmin(array_abs_idx_max_corr)]
        return lag
    else:
        return None
    

def compute_lag_one_vs_all(j, M, min_corr=0.9, max_lag=7):
    return [get_corr_lag(M[:, j], M[:, c, ], \
        min_corr=min_corr, max_lag=max_lag) \
        for c in range(M.shape[1])]


def get_corr_lag_matrix(M, min_corr=0.9, max_lag=7, n_jobs=4):

    M = array(M)
    
    n_rows, n_cols = M.shape


    pool = Pool(n_jobs)
    out = vstack(pool.map(
        partial(
            compute_lag_one_vs_all, 
            M=M, min_corr=min_corr, max_lag=max_lag
        ), 
        range(n_cols)
    ))
    pool.close()

    return out
