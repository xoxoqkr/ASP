# -*- coding: utf-8 -*-
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
test_LP_type = 'LP1'
for i in range(6):
    added_sc.append([False, 'MIP',i])
#scenarios += added_sc
random.seed(1)
start_pos2 = []
for i in range(20):
    tem = []
    for _ in range(24):
        tem.append(int(random.randrange(1,300)))
    start_pos2.append(tem)
f = open('start_pos.txt', 'a')
f.write('start record \n')
for star_info in start_pos2:
    f.write(str(star_info) + '\n')
f.close()

#input('check')
#driver_nums = list(range(10,15))
#scenarios = [[False, 'step',0],[False, 'nosubsidy',0],[True, 'MIP',0]]
driver_nums = [13,16,22]#13,16,22
#scenarios = [[False, 'MIP']]
uppers = [1500] #[1500,2000,2500,3000]
rider_coeff = []
#saved_infos = []
for customer_wait_t in [90]:
    for upper in uppers:
        for slack_time in [5]:
            for driver_num in driver_nums:
                #print('라이더 수 ',driver_num,count)
                saved_infos = []
                count1 = 0
                for ite_2 in range(10):
                    print('ite ', ite_2, count1)
                    #1 라이더 가중치 생성
                    rider_coeff_list = []
                    random.seed(count1)
                    f = open('coeff2.txt', 'a')
                    f.write('driver num'+str(driver_num)+'ITE'+str(ite_2)+';'+str(count1)+' \n')
                    for num in range(driver_num):
                        cost_coeff = random.choice([0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
                        type_coeff = 0.6 - cost_coeff  # round(random.uniform(0.8,1.2),1)
                        coeff = [cost_coeff, type_coeff, 0.4]  # [cost_coeff,type_coeff,1.5] #[cost_coeff,type_coeff,1] #[1,1,1]
                        rider_coeff_list.append(coeff)
                        f.write(str(coeff) + '\n')
                        #print(_,coeff)
                    f.write('END \n')
                    f.close()
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
                        print('시작 이후',run_type,rider_coeff_para,expected_rider_coeff_list,ite_rider_coeffs, info)
                        try:
                            print(start_pos2[ite_2])
                        except:
                            print(ite_2, len(start_pos2), start_pos2)
                            input('error')
                        #input('시나리오 확인2')

                        exec(open('Real_data_run.py', encoding='UTF8').read(),
                             globals().update(weight_update_function=info[0],subsidy_policy = info[1], saved_xlxs = saved_infos, driver_num = driver_num, upper = upper,
                                              run_type = run_type,expected_rider_coeff_list = expected_rider_coeff_list, rider_coeff_para = rider_coeff_para,
                                              rider_coeff_list = rider_coeff_list,output_rider_coeff = ite_rider_coeffs, add_file_info = str(info[2]), start_pos3 = start_pos2[ite_2],
                                              slack_time = slack_time, WEP_LP_type = test_LP_type, LT= customer_wait_t))


                        #input('확인필요')
                    excel_saved_name = ';U;{};ST;{};D;{};CW;{}'.format(upper, slack_time, driver_num, customer_wait_t)
                    DataSaver4_summary(saved_infos, saved_name= excel_saved_name)
                    #input('확인필요')
                    #
                    #input('확인필요')
                    count1 += 1