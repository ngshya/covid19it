import sys
sys.path.append(".")
import cProfile
import pstats
from covid19it.data.provincia import DataProvincia
from covid19it.utils.get_corr_lag import get_corr_lag

'''
obj_data = DataProvincia()
obj_data.load()
obj_data.parse()
'''

print(get_corr_lag([0,3,4,4,0,0], [0,0,0,2,3,3]))