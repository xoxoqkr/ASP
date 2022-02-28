# -*- coding: utf-8 -*-
import math
import numpy as np
import simpy
import Basic_class as Basic
import InstanceGen_class
import ValueRevise
import matplotlib.pyplot as plt
import random
import SubsidyPolicy_class
from ResultSave_class import DataSave
"""
global type_num
global std
global LP_type
global beta
global input_para
global input_instances
global rider_coeff
global incentive_time_ratio
global run_ite_num
global slack1
"""

global weight_update_function
global subsidy_policy

global saved_xlxs
sc_name = str(weight_update_function) + ';' + subsidy_policy
driver_num = 14
insert_thres = 1
mean_list = [0,0,0]
std_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
rider_coeff_list = []
for _ in range(driver_num):
    random.seed(1)
    cost_coeff = round(random.uniform(0.2, 0.45), 2)
    type_coeff = 0.6 - cost_coeff  # round(random.uniform(0.8,1.2),1)
    coeff = [cost_coeff, type_coeff, 0.4]  # [cost_coeff,type_coeff,1.5] #[cost_coeff,type_coeff,1] #[1,1,1]
    rider_coeff_list.append(coeff)

#weight_update_function = False # /True: 라이더 가중치 갱신 함수 작동  ; False  라이더 가중치 갱신 함수 작동X
#subsidy_policy = 'step' # 'step' : 계단형 지급 , 'MIP' : 문제 풀이, 'nosubsidy': 보조금 주지 않는 일반 상황


type_num = 4
std = 1
LP_type = 'LP2'
beta = 1
input_para = False
input_instances = None
rider_coeff = coeff
incentive_time_ratio = 0.3
run_ite_num = 1
slack1 = 1

ExpectedCustomerPreference = []
inc = int(1000/type_num)
type_fee = 0
for _ in range(type_num):
    ExpectedCustomerPreference.append(type_fee)
    type_fee += inc

# 실제 데이터 처리 부
#1 가게 데이터
store_loc_data = InstanceGen_class.LocDataTransformer('송파구store_Coor.txt', index = 0)
#2 고객 데이터
customer_loc_data = InstanceGen_class.LocDataTransformer('송파구house_Coor.txt', index = 0)
#customer_loc_data += InstanceGen_class.LocDataTransformer('송파구commercial_Coor.txt', index = len(customer_loc_data))

harversion_dist_data = np.load('rev_송파구_Haversine_Distance_data0.npy')
shortestpath_dist_data = np.load('rev_송파구_shortest_path_Distance_data0.npy')
customer_gen_numbers = 800

#파라메터
speed = 260 #meter per minute 15.6km/hr
toCenter = False
customer_wait_time = 10000
fee = None
data_dir = '데이터/new_data_2_RandomCluster.txt'
rider_intervals = InstanceGen_class.RiderGenInterval('데이터/interval_rider_data3.csv')
driver_error_pool = np.random.normal(500, 50, size=100)
customer_lamda = None
add_fee = 1500
driver_left_time = 900
driver_make_time = 800 # todo 20220228:저장되는 데이터를 관리하는 값
steps = [[]]
weight_sum = True
revise_para = True
###Running
solver_running_interval = 10
upper = 500 #todo 20220228: 보조금 상한 지급액
checker = False
print_para = True
dummy_customer_para = False #라이더가 고객을 선택하지 않는 경우의 input을 추가 하는가?


run_time = 800
validation_t = 0.7*driver_left_time
incentive_time = incentive_time_ratio * driver_left_time
ox_table = [0,0,0,0]
subsidy_offer = []
subsidy_offer_count = [0] * int(math.ceil(run_time / 60))  # [0]* (run_time//60) #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
env = simpy.Environment()
CUSTOMER_DICT = {}
RIDER_DICT = {}
start_pos = 1
#CUSTOMER_DICT[0] = Basic.Customer(env, 0, input_location=[[36, 36], [36, 36]])
env.process(InstanceGen_class.DriverMaker(env, RIDER_DICT, CUSTOMER_DICT, end_time=run_time, speed=speed,
                                          intervals=rider_intervals[0], interval_para=True, toCenter=toCenter,
                                          run_time=driver_make_time, error=np.random.choice(driver_error_pool), pref_info = 'test_rider',
                                          driver_left_time = driver_left_time, num_gen= driver_num,ExpectedCustomerPreference=ExpectedCustomerPreference,
                                          rider_coeff=rider_coeff_list, start_pos = start_pos, weight_update_function = weight_update_function))
#env.process(InstanceGen_class.CustomerGeneratorForIP(env, CUSTOMER_DICT, data_dir + '.txt', input_fee=fee, input_loc= input_para,
#                                                     type_num = type_num, std = std, input_instances= input_instances))
#env.process(InstanceGen_class.CustomerGeneratorForNPYData(env, CUSTOMER_DICT, store_loc_data, customer_loc_data,harversion_dist_data,shortestpath_dist_data,customer_gen_numbers,
#                                fee_type = 'harversion',end_time=1000,  basic_fee = 2500,customer_wait_time = 40, lamda = None, type_num = 4))
env.process(InstanceGen_class.CustomerGeneratorForNPYData2(env, CUSTOMER_DICT, harversion_dist_data,shortestpath_dist_data,customer_gen_numbers,
                                fee_type = 'harversion',end_time=1000,  basic_fee = 2500,customer_wait_time = 40, lamda = None, type_num = 4, saved_dir = data_dir))
"""
env.process(ValueRevise.SystemRunner(env, RIDER_DICT, CUSTOMER_DICT, run_time, ox_table, weight_sum = weight_sum, revise = revise_para,
                                     beta = beta, LP_type = LP_type, validation_t = validation_t,incentive_time = incentive_time, slack1 =slack1))
"""
env.process(SubsidyPolicy_class.SystemRunner(env, RIDER_DICT, CUSTOMER_DICT, run_time, interval=solver_running_interval,
                                             subsidy_offer=subsidy_offer, subsidy_offer_count=subsidy_offer_count,
                                             upper=upper,
                                             checker=checker, toCenter=toCenter,
                                             dummy_customer_para=dummy_customer_para, LP_type = LP_type, subsidy_policy = subsidy_policy))

env.run(until=run_time)
saved_data_info = DataSave(sc_name, RIDER_DICT, CUSTOMER_DICT, insert_thres, speed, run_time, subsidy_offer, subsidy_offer_count, 1, mean_list, std_list)
saved_xlxs.append([saved_data_info])
#Save_result
f = open("결과저장1209_보조금.txt", 'a')
f.write('저장 {} \n'.format('test'))

f2 = open("결과저장1209_보조금_정리.txt", 'a')
f3 = open("결과저장1209_보조금_거리차이.txt", 'a')
#f2.write('저장 {} \n'.format('test1'))
if run_ite_num == 0:
    f2.write('aplha;{};고객종류;{};거리 std;{};LP종류;{};beta;{};ValidationT;{};incentive_time_ratio;{} \n'.format(slack1,type_num,std, LP_type,beta,validation_t,incentive_time_ratio))
    f2.write('LP1;연산시간;연산횟수;obj합;LP2;연산시간;연산횟수;obj합;LP1과의 차;LP3;연산시간;연산횟수;obj합;LP1과의 차; LP1_v; LP2_v; LP3_v; '
             '총 데이터수;o1;o2;o3;11;12;13;21;22;23;31;32;33;지급보조금;LP1오차;LP2오차;LP3오차;LP1오차ABS;LP2오차ABS;LP3오차ABS\n')

for rider_name in RIDER_DICT:
    rider = RIDER_DICT[rider_name]
    f.write('고객종류;{};거리 std;{};LP종류;{};beta;{};\n'.format(type_num,std, LP_type,beta))
    f.write('w0;w1;w2;선택한 고객 수; 쌓인 데이터수;사용 데이터 수 ;실행시간;obj;euc거리;\n')
    f.write('라이더#{};org;{};{};{}; \n'.format(rider_name, rider.coeff[0],rider.coeff[1] ,rider.coeff[2]))
    print('라이더 선호',rider.coeff)
    info_name = ['LP1', 'LP2','LP3']
    count = 0
    com_t = [0, 0,0]
    obj = [0,0,0]
    for infos in [rider.LP1History, rider.LP2History, rider.LP3History]:
        print(info_name[count])
        #print(infos)
        f.write(';{};\n'.format(info_name[count]))
        for info in infos:
            #input(info)
            try:
                euc_dist = round(math.sqrt((info[0] - rider.coeff[0]) ** 2 + (info[1] - rider.coeff[1]) ** 2 + (info[2] - rider.coeff[2]) ** 2), 4)
                content = ';{};{};{};{};{};{};{};{};'.format(info[0], info[1],info[2],info[3],info[4],info[5],info[6],euc_dist)
                f.write(content + '\n')
            except:
                content = ';{};{};{};{};{};{};{};{};'.format(info[0], info[1], info[2], info[3],info[4],info[5],info[6],None)
                f.write(content+ '\n')
            #input('check1')
            com_t[count] += info[5]
            obj[count] += info[6]
            print(content)
        f.write('전체예측수;정답수;전체발생수;정답수; \n ')
        f.write('{};{};{};{}; \n '.format(ox_table[0], ox_table[1], ox_table[2],ox_table[3]))
        count += 1
    #f2정보
    euc_dist1 = round(math.sqrt(
        (rider.LP1p_coeff[0] - rider.coeff[0]) ** 2 + (rider.LP1p_coeff[1] - rider.coeff[1]) ** 2 + (rider.LP1p_coeff[2] - rider.coeff[2]) ** 2), 4)
    euc_dist2 = round(math.sqrt(
        (rider.LP2p_coeff[0] - rider.coeff[0]) ** 2 + (rider.LP2p_coeff[1] - rider.coeff[1]) ** 2 + (rider.LP2p_coeff[2] - rider.coeff[2]) ** 2), 4)
    euc_dist3 = round(math.sqrt(
        (rider.LP2p_coeff[0] - rider.LP1p_coeff[0]) ** 2 + (rider.LP2p_coeff[1] - rider.LP1p_coeff[1]) ** 2 + (
                    rider.LP2p_coeff[2] - rider.LP1p_coeff[2]) ** 2), 4)
    euc_dist4 = round(math.sqrt((rider.LP3p_coeff[0] - rider.coeff[0]) ** 2 + (rider.LP3p_coeff[1] - rider.coeff[1]) ** 2 + (rider.LP3p_coeff[2] - rider.coeff[2]) ** 2), 4)
    euc_dist5 = round(math.sqrt((rider.LP3p_coeff[0] - rider.LP1p_coeff[0]) ** 2 + (rider.LP3p_coeff[1] - rider.LP1p_coeff[1]) ** 2 + (rider.LP3p_coeff[2] - rider.LP1p_coeff[2]) ** 2), 4)

    f2_content = '{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};'.format(euc_dist1, com_t[0], len(rider.LP1History), obj[0], euc_dist2, com_t[1], len(rider.LP2History),obj[1],euc_dist3,
                                                         euc_dist4,com_t[2], len(rider.LP3History),obj[2], euc_dist5, rider.validations[0],rider.validations[1],rider.validations[2],rider.validations[3])
    f2_content += '{};{};{};{};{};{};{};{};{};{};{};{};{};'.format(rider.coeff[0],rider.coeff[1],rider.coeff[2],rider.LP1p_coeff[0],rider.LP1p_coeff[1],rider.LP1p_coeff[2],rider.LP2p_coeff[0],rider.LP2p_coeff[1],rider.LP2p_coeff[2],rider.LP3p_coeff[0],rider.LP3p_coeff[1],rider.LP3p_coeff[2],
                                                                      int(sum(rider.earn_fee)))
    if len(rider.validations_detail[0]) > 0:
        f2_content += '{};{};{};{};{};{} ;\n'.format(sum(rider.validations_detail[0])/len(rider.validations_detail[0]),sum(rider.validations_detail[1])/len(rider.validations_detail[1]),sum(rider.validations_detail[2])/len(rider.validations_detail[2]),
                                                     sum(rider.validations_detail_abs[0])/len(rider.validations_detail_abs[0]),sum(rider.validations_detail_abs[1])/len(rider.validations_detail_abs[1]),sum(rider.validations_detail_abs[2])/len(rider.validations_detail_abs[2]))
    else:
        f2_content += 'N/A;N/A;N/A;N/A;N/A;N/A ;\n'
    count2 = 0
    for info in rider.validations_detail_abs:
        f3.write('ITE;{};Count;{}; {} \n '.format(run_ite_num,count2,  info))
        count2 += 1
    f3.write('End {}\n '.format(run_ite_num))
    f2.write(f2_content)
f.close()
f2.close()
f3.close()
#데이터 확인
"""
print('고객 거리 확인')
dists = []
for ct_name in CUSTOMER_DICT:
    ct = CUSTOMER_DICT[ct_name]
    #print(ct_name,ct.location)
    dist = Basic.distance(ct.location[0],ct.location[1])
    dists.append(dist)
    print(ct_name, ct.location, dist)
plt.hist(dists, bins= 20)
plt.show()
plt.close()
"""
