EMB_DIR = './data/embD/'
STU_DIR = './data/student/'
TUTOR_DIR = './data/tutorSS/'
BENCHMARK_DIR = './data/benchmark/'
OUTPUT_DIR = './outputs/'
NEURSS_DIR = './data/neurSS/'
SYMSS_DIR = './data/symSS/'

import sys 
import time
import logging
from functools import wraps
from importlib import reload

def set_logger(log_file):
    logging.shutdown()
    reload(logging)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S', 
        filename=str(log_file),
        filemode='w'
    )
    # 
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def timeit(f):
    '''
        decorator
        time a function execution
    '''
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print ('func:%r took: %2.4f sec' % \
            (f.__name__, te-ts))
        return result
    return wrap

