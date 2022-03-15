# -*- coding: utf-8 -*-
from InstanceGen_class import InstanceGen
import random
from ResultSave_class import DataSaver4_summary
re_new_type = False  # False: 전 날의 라이더 정보를 가져 옴.  True : 매일 매일 라이더 정보를 초기화 함.



weight_update_functions = [True, False]
scenarios = [[False, 'step'],[False, 'nosubsidy'],[True, 'MIP'],[False, 'MIP']]
#scenarios = [[False, 'step'],[False, 'nosubsidy'],[False, 'MIP']]
driver_nums = list(range(10,15))
#scenarios = [[False, 'MIP']]
uppers = [1500,2000,2500,3000] #[500,1000,1500,2000]
saved_infos = []
for upper in uppers:
    count = 0
    if count == 0:
        revised_scenarios = scenarios
    else:
        revised_scenarios = scenarios[2:4]
    for driver_num in driver_nums:
        for info in revised_scenarios:
            exec(open('Real_data_run.py', encoding='UTF8').read(),
                 globals().update(weight_update_function=info[0],subsidy_policy = info[1], saved_xlxs = saved_infos, driver_num = driver_num, upper = upper))
        DataSaver4_summary(saved_infos)
    count += 1