# -*- coding: utf-8 -*-
import math
import numpy as np
import simpy
import Basic_class as Basic
import InstanceGen_class
import ValueRevise
import matplotlib.pyplot as plt
import SubsidyPolicy_class
import ResultSave_class
import copy

global type_num
global std
global LP_type
global beta
global input_para
global input_instances
global rider_coeff
global re_new_type

ExpectedCustomerPreference = []
inc = int(1000/type_num)
type_fee = 0
for _ in range(type_num):
    ExpectedCustomerPreference.append(type_fee)
    type_fee += inc


#파라메터
speed = 2
toCenter = False
customer_wait_time = 120
fee = None
data_dir = '데이터/new_data_2_RandomCluster'
rider_intervals = InstanceGen_class.RiderGenInterval('데이터/interval_rider_data3.csv')
driver_error_pool = np.random.normal(500, 50, size=100)
customer_lamda = None
driver_left_time = 120
driver_make_time = 200
steps = [[]]
driver_num = 80
weight_sum = True
revise_para = True
#필요한 인스턴스 생성
#0 시뮬레이션 환경 파라메터
run_time = 900
solver_running_interval = 10
insert_thres = 30
add_para = True
peak_times = [[0,run_time]] #결과 출력을 위한 값.
upper = 1500 #보조금 상한 지급액
checker = False
print_para = True

#1 Baisc Run에서 가져온 파라메터

wagePerHr = 9000
std_para = False
value_cal_type = 'no_return' #라이더가 주문의 가치를 계산하는 방법. 돌아오는 지점까지 계산하는지 여부
rider_start_point = [26,26]#[36,36]
#2 플랫폼 파라메터
dummy_customer_para = False #라이더가 고객을 선택하지 않는 경우의 input을 추가 하는가?
coeff_revise_option = True #True : 플랫폼이 파라메터 갱신 시도/ False : 파라메터 갱신 시도 X
zero_dist = 3
init_inc = 5
step_inc = 135
#4 시나리오 파라메터.
Problem_states = []
ITER_NUM_list =[0,0,0,0]
mean_list = [0,0,0,0]
std_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
data_dir = '데이터/new_data_2_Random' # new_data_1' ##'new_data_2_RandomCluster'

datas = [[data_dir, False,'subsidy'],[data_dir, True,'normal']] #False의 경우 보조금을 지급하는 경우, #True의 경우는 보조금을 지급하지 않는 경우



################라이더 생성##############
master_info = []
for i in datas:
    master_info.append([])

################실행 ##############
for data in datas:
    data_index = datas.index(data)
    print('check',data)
    yesterday_RIDER_DICT = []
    day_count = 0
    for ite in ITER_NUM_list:
        ###Running
        ox_table = [0,0,0,0]
        subsidy_offer = []
        subsidy_offer_count = [0] * int(math.ceil(run_time / 60))  # [0]* (run_time//60) #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        env = simpy.Environment()
        CUSTOMER_DICT = {}
        RIDER_DICT = {}
        #input('Day 시작; 라이더;{}; 고객;{}; 과거 라이더 정보; {}'.format(len(RIDER_DICT), len(CUSTOMER_DICT),len(yesterday_RIDER_DICT)))
        CUSTOMER_DICT[0] = Basic.Customer(env, 0, input_location=[[36, 36], [36, 36]])
        env.process(InstanceGen_class.DriverMaker(env, RIDER_DICT, CUSTOMER_DICT, end_time=run_time, speed=speed,
                                                  intervals=rider_intervals[0], interval_para=True, toCenter=toCenter,
                                                  run_time=driver_make_time, error=np.random.choice(driver_error_pool), pref_info = 'test_rider',
                                                  driver_left_time = driver_left_time, num_gen= driver_num,ExpectedCustomerPreference=ExpectedCustomerPreference,
                                                  rider_coeff=rider_coeff, re_new= re_new_type, day_count=day_count, yesterday_RIDER_DICT = yesterday_RIDER_DICT))
        env.process(InstanceGen_class.CustomerGeneratorForIP(env, CUSTOMER_DICT, data_dir + '.txt', input_fee=fee, input_loc= input_para,
                                                             type_num = type_num, std = std, input_instances= input_instances))
        if data[2] == 'subsidy':
            #env.process(ValueRevise.SystemRunner(env, RIDER_DICT, CUSTOMER_DICT, run_time, ox_table, weight_sum = weight_sum, revise = revise_para, beta = beta, LP_type = LP_type))
            env.process(SubsidyPolicy_class.SystemRunner(env, RIDER_DICT, CUSTOMER_DICT, run_time, interval=solver_running_interval, No_subsidy = data[1],
                                                         subsidy_offer=subsidy_offer, subsidy_offer_count = subsidy_offer_count, upper = upper,
                                                         checker= checker, toCenter = toCenter, dummy_customer_para = dummy_customer_para))
        ####### 실험 종료 후 결과 저장 ########
        env.run(until=run_time)
        info = ResultSave_class.DataSave(data, RIDER_DICT, CUSTOMER_DICT, insert_thres, speed, run_time, subsidy_offer, subsidy_offer_count, ite, mean_list, std_list)
        master_info[data_index].append(info)
        day_count += 1
        #yesterday_RIDER_DICT = copy.deepcopy(RIDER_DICT)
        #전날 라이더의 데이터를 저장하는 과정
        yesterday_RIDER_DICT = []
        for rider_name in RIDER_DICT:
            tem = []
            tem.append(copy.deepcopy(RIDER_DICT[rider_name].p_history))
            tem.append(copy.deepcopy(RIDER_DICT[rider_name].violated_choice_info))
            tem.append(copy.deepcopy(RIDER_DICT[rider_name].LP1History))
            tem.append(copy.deepcopy(RIDER_DICT[rider_name].LP2History))
            tem.append(copy.deepcopy(RIDER_DICT[rider_name].LP1p_coeff))
            tem.append(copy.deepcopy(RIDER_DICT[rider_name].LP2p_coeff))
            tem.append(copy.deepcopy(RIDER_DICT[rider_name].choice_info))
            yesterday_RIDER_DICT.append(tem)
        for rider in RIDER_DICT:
            print('목표 ;{}; -> LP1종료;{}; ->LP2종료;{};데이터수 ;{};{}'.format(RIDER_DICT[rider].coeff, RIDER_DICT[rider].LP1p_coeff,
                                                                       RIDER_DICT[rider].LP2p_coeff,len(RIDER_DICT[rider].LP1History),len(RIDER_DICT[rider].LP2History)))
        print('라이더 정보 저장 확인',len(yesterday_RIDER_DICT))
        input('Day End')
    ResultSave_class.DataSaver4_summary(master_info, saved_name="res/ITE_scenario_compete_mean_" + str(mean_list[ite]) + 'std' + str(std_list[ite]))

#Save_result
f = open("결과저장1209_보조금.txt", 'a')
f.write('저장 {} \n'.format('test'))

f2 = open("결과저장1209_보조금_정리.txt", 'a')
f2.write('저장 {} \n'.format('test1'))
f2.write('고객종류;{};거리 std;{};LP종류;{};beta;{};\n'.format(type_num,std, LP_type,beta))
f2.write(';LP1;연산시간;연산횟수;obj합;LP2;연산시간;연산횟수;obj합;LP1과의 차;\n')

for rider_name in RIDER_DICT:
    rider = RIDER_DICT[rider_name]
    f.write('고객종류;{};거리 std;{};LP종류;{};beta;{};\n'.format(type_num,std, LP_type,beta))
    f.write('w0;w1;w2;선택한 고객 수; 쌓인 데이터수;사용 데이터 수 ;실행시간;obj;euc거리;\n')
    f.write('라이더#{};org;{};{};{}; \n'.format(rider_name, rider.coeff[0],rider.coeff[1] ,rider.coeff[2]))
    print('라이더 선호',rider.coeff)
    info_name = ['LP1', 'LP2']
    count = 0
    com_t = [0, 0]
    obj = [0,0]
    for infos in [rider.LP1History, rider.LP2History]:
        print(info_name[count])
        f.write(';{};\n'.format(info_name[count]))
        for info in infos:
            #print(info)
            try:
                euc_dist = round(math.sqrt((info[0] - rider.coeff[0]) ** 2 + (info[1] - rider.coeff[1]) ** 2 + (info[2] - rider.coeff[2]) ** 2), 4)
                content = ';{};{};{};{};{};{};{};{};'.format(info[0], info[1],info[2],info[3],info[4],info[5],info[6],euc_dist)
                f.write(content + '\n')
            except:
                content = ';{};{};{};{};{};{};{};{};'.format(info[0], info[1], info[2], info[3],info[4],info[5],info[6],None)
                f.write(content+ '\n')
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
    f2_content = ';{};{};{};{};{};{};{};{};{};\n'.format(euc_dist1, com_t[0], len(rider.LP1History), obj[0], euc_dist2, com_t[1], len(rider.LP2History),obj[1],euc_dist3)
    f2.write(f2_content)
f.close()
f2.close()
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
