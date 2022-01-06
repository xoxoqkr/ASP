# -*- coding: utf-8 -*-
import math
import numpy as np
import simpy
import Basic_class as Basic
import InstanceGen_class
import ValueRevise
import matplotlib.pyplot as plt


global type_num
global std
global LP_type
global beta
global input_para
global input_instances
global rider_coeff

ExpectedCustomerPreference = []
inc = int(1000/type_num)
type_fee = 0
for _ in range(type_num):
    ExpectedCustomerPreference.append(type_fee)
    type_fee += inc


#파라메터
speed = 2
toCenter = False
customer_wait_time = 10000
fee = None
data_dir = '데이터/new_data_2_RandomCluster'
rider_intervals = InstanceGen_class.RiderGenInterval('데이터/interval_rider_data3.csv')
driver_error_pool = np.random.normal(500, 50, size=100)
customer_lamda = None
add_fee = 1500
driver_left_time = 800
driver_make_time = 200
steps = [[]]
driver_num = 1
weight_sum = True
revise_para = True
###Running
run_time = 10000

ox_table = [0,0,0,0]
subsidy_offer = []
subsidy_offer_count = [0] * int(math.ceil(run_time / 60))  # [0]* (run_time//60) #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
env = simpy.Environment()
CUSTOMER_DICT = {}
RIDER_DICT = {}
CUSTOMER_DICT[0] = Basic.Customer(env, 0, input_location=[[36, 36], [36, 36]])
env.process(InstanceGen_class.DriverMaker(env, RIDER_DICT, CUSTOMER_DICT, end_time=run_time, speed=speed,
                                          intervals=rider_intervals[0], interval_para=True, toCenter=toCenter,
                                          run_time=driver_make_time, error=np.random.choice(driver_error_pool), pref_info = 'test_rider',
                                          driver_left_time = driver_left_time, num_gen= driver_num,ExpectedCustomerPreference=ExpectedCustomerPreference,rider_coeff=rider_coeff))
env.process(InstanceGen_class.CustomerGeneratorForIP(env, CUSTOMER_DICT, data_dir + '.txt', input_fee=fee, add_fee= add_fee, steps= steps, input_loc= input_para,
                                                     type_num = type_num, std = std, input_instances= input_instances))
env.process(ValueRevise.SystemRunner(env, RIDER_DICT, CUSTOMER_DICT, run_time, ox_table, weight_sum = weight_sum, revise = revise_para, beta = beta, LP_type = LP_type))
env.run(until=run_time)

#Save_result
f = open("결과저장1209_보조금.txt", 'a')
f.write('저장 {} \n'.format('test'))

for rider_name in RIDER_DICT:
    rider = RIDER_DICT[rider_name]
    f.write('고객종류;{};거리 std;{};LP종류;{};beta;{};\n'.format(type_num,std, LP_type,beta))
    f.write('w0;w1;w2;선택한 고객 수; 쌓인 데이터수;사용 데이터 수 ;실행시간;euc거리;\n')
    f.write('라이더#{};org;{};{};{}; \n'.format(rider_name, rider.coeff[0],rider.coeff[1] ,rider.coeff[2]))
    print('라이더 선호',rider.coeff)
    info_name = ['LP1', 'LP2']
    count = 0
    for infos in [rider.LP1History, rider.LP2History]:
        print(info_name[count])
        f.write(';{};\n'.format(info_name[count]))
        for info in infos:
            #print(info)
            try:
                euc_dist = round(math.sqrt((info[0] - rider.coeff[0]) ** 2 + (info[1] - rider.coeff[1]) ** 2 + (info[2] - rider.coeff[2]) ** 2), 4)
                content = ';{};{};{};{};{};{};{};'.format(info[0], info[1],info[2],info[3],info[4],info[5],euc_dist)
                f.write(content + '\n')
            except:
                content = ';{};{};{};{};{};{};'.format(info[0], info[1], info[2], 0, 0,0,0)
                f.write(content+ '\n')
            print(content)
        f.write('전체예측수;정답수;전체발생수;정답수; \n ')
        f.write('{};{};{};{}; \n '.format(ox_table[0], ox_table[1], ox_table[2],ox_table[3]))
        count += 1


f.close()

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
