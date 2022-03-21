# -*- coding: utf-8 -*-
from InstanceGen_class import InstanceGen
import random
from ResultSave_class import DataSaver4_summary
import locale
re_new_type = False  # False: 전 날의 라이더 정보를 가져 옴.  True : 매일 매일 라이더 정보를 초기화 함.

print(locale.getpreferredencoding())


weight_update_functions = [True, False]
scenarios = [[False, 'step'],[False, 'nosubsidy'],[True, 'MIP'],[False, 'MIP']]
scenarios = [[False, 'step',0],[False, 'nosubsidy',0],[True, 'MIP',0]]
#scenarios = [[True, 'MIP',0]]
#scenarios = []
added_sc = []
for i in range(10):
    added_sc.append([False, 'MIP',i])
scenarios += added_sc
#driver_nums = list(range(10,15))
driver_nums = [13,16,22]#13,16,22
#scenarios = [[False, 'MIP']]
uppers = [1500,2000] #[1500,2000,2500,3000]
rider_coeff = []
#saved_infos = []

for upper in uppers:
    count = 0
    for driver_num in driver_nums:
        saved_infos = []
        for ite in range(30):
            #1 라이더 가중치 생성
            rider_coeff_list = []
            random.seed(count)
            for _ in range(driver_num):
                cost_coeff = random.choice([0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
                type_coeff = 0.6 - cost_coeff  # round(random.uniform(0.8,1.2),1)
                coeff = [cost_coeff, type_coeff, 0.4]  # [cost_coeff,type_coeff,1.5] #[cost_coeff,type_coeff,1] #[1,1,1]
                rider_coeff_list.append(coeff)
            ite_rider_coeffs = []
            for info in scenarios:
                print('시나리오',info)
                expected_rider_coeff_list = None
                run_type = 'else'
                rider_coeff_para = False
                print('시작 이전',run_type,rider_coeff_para,expected_rider_coeff_list,ite_rider_coeffs)
                #input('시나리오 확인1')
                if info[:2] == [False, 'MIP']:
                    if info[2] == 0:
                        run_type = 'value_revise'
                        rider_coeff_para = True
                    else:
                        expected_rider_coeff_list = ite_rider_coeffs[info[2]]
                else:
                    pass
                print('시작 이후',run_type,rider_coeff_para,expected_rider_coeff_list,ite_rider_coeffs)
                #input('시나리오 확인2')
                exec(open('Real_data_run.py', encoding='UTF8').read(),
                     globals().update(weight_update_function=info[0],subsidy_policy = info[1], saved_xlxs = saved_infos, driver_num = driver_num, upper = upper,
                                      run_type = run_type,expected_rider_coeff_list = expected_rider_coeff_list, rider_coeff_para = rider_coeff_para,
                                      rider_coeff_list = rider_coeff_list,output_rider_coeff = ite_rider_coeffs, add_file_info = str(info[2])))
            DataSaver4_summary(saved_infos)
            #input('확인필요')
            count += 1