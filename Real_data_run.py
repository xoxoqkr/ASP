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
from AssignPSolver import customers
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
global upper
global saved_xlxs
global driver_num
global run_type
global rider_coeff_list
global expected_rider_coeff_list
global output_rider_coeff
global rider_coeff_para
global add_file_info
global start_pos3
global slack_time
global WEP_LP_type
global customer_wait_t

sc_name = str(weight_update_function) + ';' + subsidy_policy + ';U;' + str(upper) + ';ST;'+str(slack_time)+';I;' + add_file_info + ';' + WEP_LP_type + ';CW;' + str(customer_wait_t)
#driver_num = 13
insert_thres = 1
mean_list = [0,0,0]
std_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

"""
rider_coeff_list = []
random.seed(1)
for _ in range(driver_num):
    #cost_coeff = round(random.uniform(0.2, 0.45), 2)
    cost_coeff = random.choice([0.2,0.25,0.3,0.35,0.4,0.45])
    #cost_coeff = 0.4
    type_coeff = 0.6 - cost_coeff  # round(random.uniform(0.8,1.2),1)
    coeff = [cost_coeff, type_coeff, 0.4]  # [cost_coeff,type_coeff,1.5] #[cost_coeff,type_coeff,1] #[1,1,1]
    rider_coeff_list.append(coeff)
"""


#weight_update_function = False # /True: 라이더 가중치 갱신 함수 작동  ; False  라이더 가중치 갱신 함수 작동X
#subsidy_policy = 'step' # 'step' : 계단형 지급 , 'MIP' : 문제 풀이, 'nosubsidy': 보조금 주지 않는 일반 상황


type_num = 4
std = 1
LP_type = 'LP3'
beta = 1
input_para = False
input_instances = None
#rider_coeff = coeff
incentive_time_ratio = 0.3
run_ite_num = 1
slack1 = 1

ExpectedCustomerPreference = []
inc = int(1000/type_num)
type_fee = 0
for _ in range(type_num):
    type_fee += inc
    ExpectedCustomerPreference.append(type_fee)

# 실제 데이터 처리 부
#1 가게 데이터
store_loc_data = InstanceGen_class.LocDataTransformer('송파구store_Coor.txt', index = 0)
#2 고객 데이터
customer_loc_data = InstanceGen_class.LocDataTransformer('송파구house_Coor.txt', index = 0)
#customer_loc_data += InstanceGen_class.LocDataTransformer('송파구commercial_Coor.txt', index = len(customer_loc_data))

harversion_dist_data = np.load('data_rev/송파구_test0311_Haversine_Distance_data3.npy')
shortestpath_dist_data = np.load('data_rev/송파구_test0311_shortest_path_Distance_data3.npy')
customer_gen_numbers = 350

#파라메터
speed = 260 #meter per minute 15.6km/hr
toCenter = False
customer_wait_time = customer_wait_t
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
solver_running_interval = 5
sc_name += 'SI;' + str(solver_running_interval) + ';'
#upper = 10000 #todo 20220228: 보조금 상한 지급액
checker = False
print_para = True
dummy_customer_para = False #라이더가 고객을 선택하지 않는 경우의 input을 추가 하는가?


run_time = 800
validation_t = 0
incentive_time = 0 # incentive_time_ratio * driver_left_time
ox_table = [0,0,0,0]
subsidy_offer = []
subsidy_offer_count = [0] * int(math.ceil(run_time / 60))  # [0]* (run_time//60) #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
env = simpy.Environment()
CUSTOMER_DICT = {}
RIDER_DICT = {}
#start_pos = 1
#CUSTOMER_DICT[0] = Basic.Customer(env, 0, input_location=[[36, 36], [36, 36]])
env.process(InstanceGen_class.DriverMaker(env, RIDER_DICT, CUSTOMER_DICT, end_time=run_time, speed=speed,
                                          intervals=rider_intervals[0], interval_para=True, toCenter=toCenter,
                                          run_time=driver_make_time, error=np.random.choice(driver_error_pool), pref_info = 'test_rider',
                                          driver_left_time = driver_left_time, num_gen= driver_num,ExpectedCustomerPreference=ExpectedCustomerPreference,
                                          rider_coeff=rider_coeff_list, start_pos = start_pos3, weight_update_function = weight_update_function,
                                          exp_rider_coeff = expected_rider_coeff_list, subsidyForweight = rider_coeff_para, LP_type=WEP_LP_type))
#env.process(InstanceGen_class.CustomerGeneratorForIP(env, CUSTOMER_DICT, data_dir + '.txt', input_fee=fee, input_loc= input_para,
#                                                     type_num = type_num, std = std, input_instances= input_instances))
#env.process(InstanceGen_class.CustomerGeneratorForNPYData(env, CUSTOMER_DICT, store_loc_data, customer_loc_data,harversion_dist_data,shortestpath_dist_data,customer_gen_numbers,
#                                fee_type = 'harversion',end_time=1000,  basic_fee = 2500,customer_wait_time = 40, lamda = None, type_num = 4))
env.process(InstanceGen_class.CustomerGeneratorForNPYData2(env, CUSTOMER_DICT, harversion_dist_data,shortestpath_dist_data,customer_gen_numbers,
                                fee_type = 'harversion',end_time=1000,  basic_fee =2500,customer_wait_time = customer_wait_time, lamda = None, type_num = type_num, saved_dir = data_dir))
if run_type == 'value_revise:':
    #env.process(ValueRevise.SystemRunner(env, RIDER_DICT, CUSTOMER_DICT, run_time, ox_table, weight_sum = weight_sum, revise = revise_para,
    #                                     beta = beta, LP_type = LP_type, validation_t = validation_t,incentive_time = incentive_time, slack1 =slack1))
    pass
else:
    env.process(SubsidyPolicy_class.SystemRunner(env, RIDER_DICT, CUSTOMER_DICT, run_time, interval=solver_running_interval,
                                                 subsidy_offer=subsidy_offer, subsidy_offer_count=subsidy_offer_count,
                                                 upper=upper,
                                                 checker=checker, toCenter=toCenter,
                                                 dummy_customer_para=dummy_customer_para, LP_type = WEP_LP_type, subsidy_policy = subsidy_policy,
                                                 slack_time = slack_time))



env.run(until=run_time)
print('고객 거리 확인')
dists = []
f_c = open(sc_name + 'ctinfo.txt','a')
f_c.write('{};{}; \n'.format(sc_name, driver_num))
for ct_name in CUSTOMER_DICT:
    ct = CUSTOMER_DICT[ct_name]
    #print(ct_name,ct.location)
    dist = Basic.distance(ct.location[0],ct.location[1])
    dists.append(dist)
    print(ct_name, ct.location, dist, ct.type)
    ct_t = ct.time_info
    test_c = '{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};'.format(ct.name, ct.location[0], ct.location[1], ct.type, ct_t[0],ct_t[1],ct_t[2],ct_t[3],ct_t[4],ct.fee[0],ct.fee[1],ct.fee[2],ct.fee[3],ct.fee_t, Basic.distance(ct.location[0], ct.location[1]))
    #input('확인1')
    try:
        min_val = min(ct.fee_history)
        test_c += (str(min_val) + ';')
    except:
        test_c += 'None;'
    test_c += str(ct.fee_history) + '\n'
    #input('확인2')
    f_c.write(test_c)
f_c.write('End \n')
f_c.close()

#plt.hist(dists, bins= 20)
#plt.show()
#plt.close()

#input('확인')

saved_data_info = DataSave(sc_name, RIDER_DICT, CUSTOMER_DICT, insert_thres, speed, run_time, subsidy_offer, subsidy_offer_count, 1, mean_list, std_list)
ave_euc_error = []
for rider_name in RIDER_DICT:
    c = RIDER_DICT[rider_name].coeff
    if WEP_LP_type == 'LP1':
        c1 = RIDER_DICT[rider_name].LP1p_coeff
    else:
        c1 = RIDER_DICT[rider_name].LP3p_coeff
    val = float(math.sqrt((c[0] - c1[0])**2 + (c[1] - c1[1])**2 + (c[2] - c1[2])**2))
    ave_euc_error.append(val)
try:
    ave_euc_error1 = np.mean(ave_euc_error)
except:
    ave_euc_error1 = 'None'
saved_data_info.append(ave_euc_error1)
f = open("Rider_income"+ sc_name +".txt", 'a')
f.write('start; '+str(driver_num)+ ';\n')
f.write('name; r_earing; \n')
earn_data = ['//']
for rider_name in RIDER_DICT:
    r_earing = RIDER_DICT[rider_name].total_earn
    content = '{};{}; \n'.format(rider_name, sum(r_earing))
    earn_data.append(sum(r_earing))
    f.write(content)
f.close()
#보조금 지급 고객 정보
kpi1 = 0
kpi2 = 0
kpi3 = 0
kpi4 = []
kpi5 = []
for name4 in CUSTOMER_DICT:
    ct = CUSTOMER_DICT[name4]
    if len(ct.fee_history) > 0:
        if ct.time_info[4] == None:
            kpi2 += 1
        else:
            kpi1 += 1
    if ct.fee[1] > 0 and ct.time_info[4] != None:
        kpi3 += 1
    if ct.fee[1] > 0:
        try:
            kpi5.append(max(ct.fee_history) - min(ct.fee_history))
        except:
            kpi5.append(0)
        if subsidy_policy == 'step':
            kpi4.append((ct.fee[1]/ct.fee[0])*10)
        else:
            kpi4.append(len(ct.fee_history))


if kpi1 + kpi2 == 0:
    add_data2 = ['//',None,None]
else:
    add_data2 = ['//',kpi1 / (kpi1 + kpi2), kpi3 / (kpi1 + kpi2)]
try:
    add_data2.append(sum(kpi4)/len(kpi4))
except:
    add_data2.append(None)
try:
    add_data2.append(sum(kpi5) / len(kpi5))
except:
    add_data2.append(None)

saved_data_info += add_data2
saved_data_info += earn_data


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
    #input('라이더 {} 종료 '.format(rider.name))
    f.write('ORG : {} ; LP3 : {};\n'.format(rider.coeff, rider.LP3p_coeff))
    #f2정보
    print('coeff확인',rider.coeff, rider.LP1p_coeff, rider.LP2p_coeff, rider.LP3p_coeff)
    #input('종료확인1')
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
#input('종료 확인2')
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
ave_dist = []
fee_dist = []
for customer_name in CUSTOMER_DICT:
    ct = CUSTOMER_DICT[customer_name]
    if ct.done == True:
        ave_dist.append(Basic.distance(ct.location[0], ct.location[1]))
        fee_dist.append(ct.fee_base_dist)
print('서비스된 고객 수 {} : 평균 거리 {} : 수수료 기준 거리 {}'.format(len(ave_dist), sum(ave_dist)/len(ave_dist), sum(fee_dist)/len(fee_dist)))

ite_range = range(0,30,5)
if rider_coeff_para == True:
    f = open("COEFF Check_org.txt", 'a')
    for ite_ in ite_range:
        tem = []
        for rider_name in RIDER_DICT:
            try:
                #ite1 = min(len(RIDER_DICT[rider_name].LP3History)-1, ite_)
                #tem.append(RIDER_DICT[rider_name].LP3History[ite1][:3])
                ite1 = min(len(RIDER_DICT[rider_name].LP1History) - 1, ite_)
                tem.append(RIDER_DICT[rider_name].LP1History[ite1][:3])
            except:
                #tem.append(RIDER_DICT[rider_name].LP3p_coeff[-1][:3])
                tem.append(RIDER_DICT[rider_name].LP1p_coeff[-1][:3])
        output_rider_coeff.append(tem)
        f.write(str(ite_) + 'start1 \n')
        f.write(str(tem) + '\n')
    f.write('원 자료 저장 \n')
    f.close()

f = open("COEFF Check.txt", 'a')
f.write('start \n')
for rider_name in RIDER_DICT:
    c = RIDER_DICT[rider_name].coeff
    c1 = RIDER_DICT[rider_name].LP3p_coeff
    content = 'Rider;{};coeff;{};{};{};LP3;{};{};{}; \n'.format(rider_name, c[0],c[1],c[2],c1[0],c1[1],c1[2])
    f.write(content)
f.close()

