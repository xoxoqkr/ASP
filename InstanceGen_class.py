# -*- coding: utf-8 -*-
#.py 설명 :
# ->InstanceGen_class.py : 고객, 라이더 생성 관련 함수
import Basic_class as Basic
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import copy

def InstanceGen(unit_dist= 8, std = 1, max_x = 50, max_y = 50, gen_n = 100):
    coordinates = []
    pool = np.random.normal(unit_dist, std, size= 10000)
    """
    plt.hist(pool, bins=20)
    plt.show()
    plt.close()
    input('pool 확인')    
    """
    for counnt in range(gen_n):
        for try_num in range(1000):
            store_loc = [random.randrange(0, int(max_x / 2)), random.randrange(0, int(max_y / 2))]
            req_dist = random.choice(pool)
            angle = math.radians(random.randrange(0,360))
            customer_loc = [store_loc[0] + round(req_dist*math.cos(angle),4),store_loc[1] + round(req_dist*math.sin(angle),4) ]
            #customer_loc = [store_loc[0] + round(req_dist*math.sin(angle),4),store_loc[0] + round(req_dist*math.cos(angle),4) ]
            if 0 <= customer_loc[0] <= max_x and 0 <= customer_loc[1] <= max_y:
                coordinates.append([store_loc, customer_loc])
                break
    return coordinates


def FeeCalculator(distance, steps):
    """
    주어진 거리에 대한 플랫폼의 추가 지급 수수료 금액을 계산.
    :param distance: 이동 거리
    :param steps: 계단형 정보 [[시작 거리, 종료 거리, 수수료],...,]
    :return:
    """
    for step in steps:
        if step[0] <= distance < step[1]:
            return step[2]



def CustomerGeneratorForIP(env, customer_dict, dir=None, end_time=1000,  customer_wait_time = 40, input_fee = None, lamda = None, input_loc = False,
                           unit_dist = 8, std = 1, max_x = 50, max_y = 50, type_num = 4, input_instances = None):
    """
    주어진 입력 값에 대한 고객을 생성. 아래 요인들이 중요
    -배송비 책정
    -고객 생성 간격
    -고객 위치 분포

    :param env: simpy Environment
    :param customer_dict: 고객 dict <- 생성된 고객이 저장되는 dict
    :param dir: 고객 데이터 저장 위치
    :param end_time: 고객 생성 종료 시점
    :param customer_wait_time: 고객의 주문 발생 후 기다리는 시간. 이 시간 동안 서비스를 받지 못한 고객은 주문을 철회
    :param fee: 배송 요금(fee == None인 경우에는 거리에 대한 계단형 방식-FeeCalculator-으로 요금 계산)
    :param lamda: 고객 생성 간격
    :param add_fee: 추가 요금
    :param basic_fee: 기본 요금
    :param steps: 거리에 비례해 계산되는 요금을 계산하기 위한 거리별 요금 정보
    """
    datas = open(dir, 'r')
    lines = datas.readlines()
    if input_instances == None:
        coordinates = InstanceGen(unit_dist=unit_dist, std=std, max_x=max_x, max_y=max_y, gen_n=10000)
    else:
        coordinates = input_instances
    for line in lines[3:]:
        data = line.split(',')
        store_loc = [float(data[1]), float(data[2])]
        customer_loc = [float(data[3]), float(data[4])]
        if input_fee == None:
            #distance = int(Basic.distance(store_loc, customer_loc))
            #fee = FeeCalculator(distance, steps) + basic_fee #기본 수수료(basic_fee)에 거리마다 계단 형으로 추가 요금이 발생하는 요금제
            fee = round((Basic.distance(store_loc, customer_loc) / 10) * 100,2) + 2500  # 2500
        else:
            fee = fee
        if input_loc != False:
            store_loc = coordinates[int(data[0])][0]
            customer_loc = coordinates[int(data[0])][1]
        c = Basic.Customer(env, int(data[0]), input_location = [store_loc, customer_loc], fee= fee, end_time= customer_wait_time, far = int(data[6]),type_num = type_num)
        customer_dict[int(data[0])] = c
        #print('Time',round(env.now,2) ,'CT#', c.name, 'gen')
        if lamda == None:
            yield env.timeout(float(data[7]))
        else:
            try:
                yield env.timeout(Basic.NextMin(lamda))
            except:
                print("lamda require type int+ or float+")
                yield env.timeout(4)
        if env.now > end_time:
            break


def CustomerGeneratorForNPYData(env, customer_dict, store_loc_data, customer_loc_data, harversion_dist_data,shortestpath_dist_data,gen_numbers,
                                fee_type = 'harversion',end_time=1000,  basic_fee = 2500,customer_wait_time = 40, lamda = None, type_num = 4):
    """
    주어진 입력 값에 대한 고객을 생성. 아래 요인들이 중요
    -배송비 책정
    -고객 생성 간격
    -고객 위치 분포

    :param env: simpy Environment
    :param customer_dict: 고객 dict <- 생성된 고객이 저장되는 dict
    :param dir: 고객 데이터 저장 위치
    :param end_time: 고객 생성 종료 시점
    :param customer_wait_time: 고객의 주문 발생 후 기다리는 시간. 이 시간 동안 서비스를 받지 못한 고객은 주문을 철회
    :param fee: 배송 요금(fee == None인 경우에는 거리에 대한 계단형 방식-FeeCalculator-으로 요금 계산)
    :param lamda: 고객 생성 간격
    :param add_fee: 추가 요금
    :param basic_fee: 기본 요금
    :param steps: 거리에 비례해 계산되는 요금을 계산하기 위한 거리별 요금 정보
    """
    ordered_customer_names = []
    for num in range(1,gen_numbers):
        store_name = num
        customer_name = num


        store_name = random.choice(range(len(store_loc_data)))
        count = 0



        while count < 1000:
            customer_name = random.choice(range(len(customer_loc_data)))
            count += 1
            if customer_name not in ordered_customer_names and harversion_dist_data[store_name,customer_name] > 0 and shortestpath_dist_data[store_name,customer_name] > 0:
                break
        ordered_customer_names.append(customer_name)
        store_loc = store_loc_data[store_name][0]
        customer_loc = customer_loc_data[customer_name][0]
        if fee_type == 'shortest_path':
            ODdist = harversion_dist_data[store_name,customer_name]
        else:
            ODdist = shortestpath_dist_data[store_name,customer_name]
        fee = round((ODdist*10) * 100, 2) + basic_fee  # 2500 #ODdist 는 키로미터
        input('거리{} 수수료{}'.format(ODdist,fee))
        far_para = 0
        if ODdist >= 3:
            far_para = 1
        c = Basic.Customer(env, num, input_location=[store_loc, customer_loc], fee=fee,
                           end_time=customer_wait_time, far=far_para, type_num=type_num)
        customer_dict[num] = c
        if lamda == None:
            yield env.timeout(3)
        else:
            try:
                yield env.timeout(Basic.NextMin(lamda))
            except:
                print("lamda require type int+ or float+")
                yield env.timeout(4)
        if env.now > end_time:
            break


def CustomerGeneratorForNPYData2(env, customer_dict, harversion_dist_data,shortestpath_dist_data,gen_numbers,
                                fee_type = 'harversion',end_time=1000,  basic_fee = 2500,customer_wait_time = 40, lamda = None, type_num = 4, saved_dir = None):
    """
    주어진 입력 값에 대한 고객을 생성. 아래 요인들이 중요
    -배송비 책정
    -고객 생성 간격
    -고객 위치 분포

    :param env: simpy Environment
    :param customer_dict: 고객 dict <- 생성된 고객이 저장되는 dict
    :param dir: 고객 데이터 저장 위치
    :param end_time: 고객 생성 종료 시점
    :param customer_wait_time: 고객의 주문 발생 후 기다리는 시간. 이 시간 동안 서비스를 받지 못한 고객은 주문을 철회
    :param fee: 배송 요금(fee == None인 경우에는 거리에 대한 계단형 방식-FeeCalculator-으로 요금 계산)
    :param lamda: 고객 생성 간격
    :param add_fee: 추가 요금
    :param basic_fee: 기본 요금
    :param steps: 거리에 비례해 계산되는 요금을 계산하기 위한 거리별 요금 정보
    """
    lamdas = []
    if saved_dir != None:
        datas = open(saved_dir, 'r')
        lines = datas.readlines()
        for line in lines[3:]:
            data = line.split(',')
            lamdas.append(float(data[7]))
            #input('데이터 간격{}'.format(data[7]))
    else:
        for _ in range(gen_numbers):
            lamdas.append(3)
    for num in range(0,gen_numbers):
        if fee_type == 'shortest_path':
            ODdist = shortestpath_dist_data[num,num]
        else:
            ODdist = harversion_dist_data[num,num]
        fee = round(int(ODdist*10) * 100, 2) + basic_fee  # 2500 #ODdist 는 키로미터
        print('거리{} 수수료{}'.format(ODdist,fee))
        far_para = 0
        if ODdist >= 3:
            far_para = 1
        c = Basic.Customer(env, num, input_location=[num, num], fee=fee,
                           end_time=customer_wait_time, far=far_para, type_num=type_num)
        customer_dict[num] = c
        yield env.timeout(lamdas[num])
        """
        if lamdas == None:
            yield env.timeout(3)
        else:
            try:
                yield env.timeout(Basic.NextMin(lamda))
            except:
                print("lamda require type int+ or float+")
                yield env.timeout(4)        
        """
        if env.now > end_time:
            break

def LocDataTransformer(dir, index = 0):
    res = []
    datas = open(dir, 'r')
    lines = datas.readlines()
    for line in lines[2:]:
        data = line.split(';')[:4]
        info = [index, int(data[1]), [float(data[2][1:len(data[2])-1].split(',')[0]),float(data[2][1:len(data[2])-1].split(',')[1])], int(data[3])]
        res.append(info)
        index += 1
    datas.close()
    return res


def DriverMaker(env, driver_dict, customer_set ,speed = 2, end_time = 800, intervals = [], interval_para = False, interval_res = [],
                toCenter = True, error = 0, run_time = 900, pref_info = None, driver_left_time = 120, print_para = False,
                start_pos = [26,26], value_cal_type = 'return', num_gen = 10, coeff_revise_option = False, weight_sum = False,
                ExpectedCustomerPreference = [0,250,500,750], rider_coeff = None, re_new = True, day_count = 0, yesterday_RIDER_DICT = None,
                weight_update_function = True):
    """
    주어진 입력값으로 행동하는 라이더를 생성
    :param env: simpy Environment
    :param driver_dict: 라이더 dict <- 생성된 라이더가 저장되는 dict
    :param customer_set: 고객 dict <- 생성된 고객이 저장되는 dict
    :param speed: 차량 속도
    :param end_time: 라이더 발생 종료 시점
    :param intervals: 라이더 발생 간격
    :param interval_para:
    :param interval_res: 라이더 발생 간격 저장소
    :param toCenter: PriorityOrdering에서 라이더의 가치를 판단하는 방식(toCenter == True시 라이더가 다시 center로 돌아가는 것 까지 운행 비용에 고려함)
    :param error: 라이더에 대해서 플랫폼이 가지는 error
    :param run_time: 시뮬레이션 종료 시점
    :param pref_info: PriorityOrdering에서 가치 함수 계산에 사용되는 방법
    :param driver_left_time: 라이더가 플랫폼에 들어온 이후 일을 하는 시간(default = 120)
    :param print_para: 라이더 print 구문 파라메터 (T: 프린트문 실행, F : 프린트문 실행X)
    :param start_pos: 라이더 처음 발생 위치
    :param value_cal_type: 고객의 가치를 계산하는 방식
    """
    name = 0
    while env.now < end_time and name < num_gen:
        rider = Basic.Rider(env, name, speed, customer_set, toCenter = toCenter, error = error, run_time = run_time,
                            pref_info= pref_info, left_time=driver_left_time, print_para = print_para, start_pos= start_pos,
                            value_cal_type = value_cal_type, coeff_revise_option = coeff_revise_option, weight_sum = weight_sum,
                            ExpectedCustomerPreference = ExpectedCustomerPreference)
        #input('day_count {}, renew{}, yesterday_RIDER_DICT {}'.format(day_count, re_new,yesterday_RIDER_DICT))
        if day_count > 0 and re_new == False and yesterday_RIDER_DICT != None:
            #input('라이더 갱신 시도')
            try:
                rider.p_history = yesterday_RIDER_DICT[name][0]
                rider.violated_choice_info = yesterday_RIDER_DICT[name][1]
                rider.LP1History = yesterday_RIDER_DICT[name][2]
                rider.LP2History = yesterday_RIDER_DICT[name][3]
                rider.LP1p_coeff = yesterday_RIDER_DICT[name][4]
                rider.LP2p_coeff = yesterday_RIDER_DICT[name][5]
                rider.choice_info = yesterday_RIDER_DICT[name][6]
                rider.coeff = yesterday_RIDER_DICT[name][7]
                #input('라이더 갱신 성공')
            except:
                input('라이더 갱신 실패')
        rider.weight_update_function = weight_update_function
        if rider_coeff != None:
            rider.coeff = rider_coeff[name]
        driver_dict[name] = rider
        if interval_para == False:
            #print('Hr',intervals[int(env.now//60)])
            next_time = Basic.NextMin(intervals[int(env.now//60)])
            interval_res.append(next_time)
        else:
            if name >= len(intervals):
                break
            next_time = intervals[name]
        name += 1
        print('T {} rider {} Gen'.format(env.now,rider.name) )
        yield env.timeout(next_time)


def RiderGenInterval(dir, lamda = None):
    """
    dir에서 자료를 읽어와 라이더의 interval 생성
    만약 lamda에 값이 할당되면, 시간당 lamda를 가지는 포아송 과정을 발생시켜,
    라이더 발생 interval을 발생시킨다.
    :param dir: 자료 directory
    :param lamda: 시간 당 라이더 발생 수
    :return: 라이더 interval list
    """
    rider_intervals = []
    if lamda == None:
        #rider gen lamda 읽기.
        datas = open(dir, 'r')
        lines = datas.readlines()
        for line in lines[1:]:
            interval = line.split(',')
            interval = interval[1:len(interval)]
            tem = []
            for i in interval:
                # print(i)
                # input('STOP')
                tem.append(float(i))
            rider_intervals.append(tem)
    else:
        for _ in range(1000):
            rider_intervals.append(Basic.NextMin(lamda))
    return rider_intervals
