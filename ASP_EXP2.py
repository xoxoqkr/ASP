# -*- coding: utf-8 -*-
from InstanceGen_class import InstanceGen
import random
from ResultSave_class import DataSaver4_summary
re_new_type = False  # False: 전 날의 라이더 정보를 가져 옴.  True : 매일 매일 라이더 정보를 초기화 함.



weight_update_functions = [True, False]
#scenarios = [[False, 'step'],[True, 'MIP'],[False, 'MIP'],[False, 'nosubsidy']]
scenarios = [[True, 'MIP'],[False, 'MIP']]
driver_nums = list(range(15,16))
#scenarios = [[True, 'MIP']]
saved_infos = []
for driver_num in driver_nums:
    for info in scenarios:
        exec(open('Real_data_run.py', encoding='UTF8').read(),
             globals().update(weight_update_function=info[0],subsidy_policy = info[1], saved_xlxs = saved_infos, driver_num = driver_num))
    DataSaver4_summary(saved_infos)