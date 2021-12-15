# -*- coding: utf-8 -*-
import math
import numpy as np
import simpy
import Basic_class as Basic
import InstanceGen_class
import ValueRevise


#파라메터
speed = 2
toCenter = True
customer_wait_time = 90
fee = None
data_dir = '데이터/new_data_2_RandomCluster'
rider_intervals = InstanceGen_class.RiderGenInterval('데이터/interval_rider_data3.csv')
driver_error_pool = np.random.normal(500, 50, size=100)
customer_lamda = None
add_fee = 1500
driver_left_time = 800
driver_make_time = 200
steps = [[]]
driver_num = 3
weight_sum = True

revise_para = True
###Running
run_time = 900

ox_table = [0,0,0,0]
subsidy_offer = []
subsidy_offer_count = [0] * int(math.ceil(run_time / 60))  # [0]* (run_time//60) #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
env = simpy.Environment()
CUSTOMER_DICT = {}
RIDER_DICT = {}
CUSTOMER_DICT[0] = Basic.Customer(env, 0, input_location=[[36, 36], [36, 36]])
env.process(InstanceGen_class.DriverMaker(env, RIDER_DICT, CUSTOMER_DICT, end_time=run_time, speed=speed,
                                          intervals=rider_intervals[0], interval_para=True, toCenter=toCenter,
                                          run_time=driver_make_time, error=np.random.choice(driver_error_pool), pref_info = 'test_rider', driver_left_time = driver_left_time, num_gen= driver_num))
env.process(InstanceGen_class.CustomerGeneratorForIP(env, CUSTOMER_DICT, data_dir + '.txt', input_fee=fee, add_fee= add_fee, steps= steps))
env.process(ValueRevise.SystemRunner(env, RIDER_DICT, CUSTOMER_DICT, run_time, ox_table, weight_sum = weight_sum, revise = revise_para))
env.run(until=run_time)

#Save_result
f = open("결과저장1209_보조금.txt", 'a')
f.write('저장 {} \n'.format('test'))

for rider_name in RIDER_DICT:
    rider = RIDER_DICT[rider_name]
    f.write('라이더#{};org;{};{};{}; \n'.format(rider_name, rider.coeff[0],rider.coeff[1] ,rider.coeff[2]))
    for info in rider.p_history:
        f.write(';{};{};{}; \n '.format(info[0], info[1],info[2]))
    f.write('전체예측수;정답수;전체발생수;정답수; \n ')
    f.write('{};{};{};{}; \n '.format(ox_table[0], ox_table[1], ox_table[2],ox_table[3]))
f.close()