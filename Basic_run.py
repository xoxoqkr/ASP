# -*- coding: utf-8 -*-
#.py 설명 :
# -> Basic_run.py : 보조금 지급과 보조금 지급X 시뮬레이션 실행 부

import simpy
import Basic_class
import SubsidyPolicy_class
import InstanceGen_class
import ResultSave_class
import numpy as np
import math

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
#1 라이더 관련 파라메터
speed = 2.2
wagePerHr = 9000
toCenter = False
std_para = False
value_cal_type = 'no_return' #라이더가 주문의 가치를 계산하는 방법. 돌아오는 지점까지 계산하는지 여부
rider_start_point = [26,26]#[36,36]
#2 플랫폼 파라메터
dummy_customer_para = False #라이더가 고객을 선택하지 않는 경우의 input을 추가 하는가?
driver_error_pool = np.random.normal(500, 50, size=100)
basic_fee = 2500
steps = []
zero_dist = 3
init_inc = 5
step_inc = 135
for i in range(11):
    if i < zero_dist:
        steps.append([i*2 ,(i+1)*2,0])
    elif i == zero_dist:
        steps.append([i * 2, (i + 1) * 2, init_inc])
    else:
        steps.append([i * 2, (i + 1) * 2, init_inc + (i - zero_dist)*step_inc])
#3 고객 파라메터
max_food_in_bin_time = 20
customer_wait_time = 80
fee = None #Basic.distance(store_loc, customer_loc)*120 + 3500 -> 이동거리*120 + 기본료(3500)
#4 시나리오 파라메터.
Problem_states = []
ITER_NUM_list =[0]
mean_list = [0,0,0]
std_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
data_dir = '데이터/new_data_2_Random' # new_data_1' ##'new_data_2_RandomCluster'

datas = [[data_dir, False,'subsidy'],[data_dir, True,'normal']] #False의 경우 보조금을 지급하는 경우, #True의 경우는 보조금을 지급하지 않는 경우



################라이더 생성##############

rider_intervals = InstanceGen_class.RiderGenInterval('데이터/interval_rider_data3.csv')
master_info = []
for i in datas:
    master_info.append([])

################실행 ##############
for data in datas:
    data_index = datas.index(data)
    print('check',data)
    for ite in ITER_NUM_list:
        #####Runiing part######
        subsidy_offer = []
        subsidy_offer_count = [0]*int(math.ceil(run_time/60)) #[0]* (run_time//60) #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        env = simpy.Environment()
        CUSTOMER_DICT = {}
        RIDER_DICT = {}
        CUSTOMER_DICT[0] = Basic_class.Customer(env, 0, input_location=[rider_start_point, rider_start_point])
        env.process(InstanceGen_class.DriverMaker(env, RIDER_DICT, CUSTOMER_DICT, end_time=run_time, speed=speed, intervals= rider_intervals[0],
                                                  interval_para= True, toCenter = toCenter, run_time= run_time, error= np.random.choice(driver_error_pool),
                                                  print_para = print_para, start_pos = rider_start_point, value_cal_type = value_cal_type))
        env.process(InstanceGen_class.CustomerGeneratorForIP(env, CUSTOMER_DICT, data[0] + '.txt', customer_wait_time=customer_wait_time, fee = fee, basic_fee = basic_fee, steps = steps))
        env.process(SubsidyPolicy_class.SystemRunner(env, RIDER_DICT, CUSTOMER_DICT, run_time, interval=solver_running_interval, No_subsidy = data[1],
                                                     subsidy_offer=subsidy_offer, subsidy_offer_count = subsidy_offer_count, upper = upper,
                                                     checker= checker, toCenter = toCenter, dummy_customer_para = dummy_customer_para))
        env.run(until=run_time)
        ####### 실험 종료 후 결과 저장 ########
        info = ResultSave_class.DataSave(data, RIDER_DICT, CUSTOMER_DICT, insert_thres, speed, run_time, subsidy_offer, subsidy_offer_count, ite, mean_list, std_list)
        master_info[data_index].append(info)
ResultSave_class.DataSaver4_summary(master_info, saved_name="res/ITE_scenario_compete_mean_" + str(mean_list[ite]) + 'std' + str(std_list[ite]))
