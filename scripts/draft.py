import sys
sys.path.append(".")
import cProfile
import pstats
from covid19it.data.provincia import DataProvincia

obj_data = DataProvincia()
obj_data.load()
obj_data.parse()
