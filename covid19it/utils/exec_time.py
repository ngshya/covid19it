import time
from functools import wraps
from .logger import logger

def exec_time(func):
    '''
    Log execution time of the input function.
    
    Parameters
    ----------
    func : function
        The function to be decorated.
    
    Returns
    -------
    function
        Decorated function.
    '''

    @wraps(func)
    def wrapper_func(*args, **kwargs):
        str_func_name = func.__name__
        # logging.debug("%s START" % (str_func_name))
        tms_start = time.time()
        output = func(*args, **kwargs)
        tms_end = time.time()
        # logging.debug("%s END" % (str_func_name))
        logger.debug("%s execution time %.4f seconds" % \
            (str_func_name, tms_end-tms_start))
        return output

    return wrapper_func