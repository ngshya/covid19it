from numpy import array, abs, mean


def MMR(y_real, y_pred, range_min, range_max):
    y_real = array(y_real)
    y_pred = array(y_pred)
    assert len(y_real) <= len(y_pred)
    y_pred = y_pred[:len(y_real)]
    return  (mean(abs(y_pred - y_real)[range_min:range_max]) / mean(abs(y_real)[range_min:range_max]))