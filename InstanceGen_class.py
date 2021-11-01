# -*- coding: utf-8 -*-
#.py 설명 :
# ->InstanceGen_class.py : 고객, 라이더 생성 관련 함수
import Basic_class as Basic


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



def CustomerGeneratorForIP(env, customer_dict, dir=None, end_time=1000,  customer_wait_time = 40, input_fee = None, lamda = None, add_fee = 0, basic_fee = 3500, steps = []):
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
    for line in lines[3:]:
        data = line.split(',')
        #print(data)
        #print(float(data[1]),float(data[2]),float(data[3]),float(data[4]),float(data[7]))
        #input('Stop')
        store_loc = [float(data[1]), float(data[2])]
        customer_loc = [float(data[3]), float(data[4])]
        if input_fee == None:
            #distance = int(Basic.distance(store_loc, customer_loc))
            #input('{} : {} : {}'.format(FeeCalculator(distance, steps), steps, distance))
            #fee = FeeCalculator(distance, steps) + basic_fee #기본 수수료(basic_fee)에 거리마다 계단 형으로 추가 요금이 발생하는 요금제
            #print('거리 {}'.format(Basic.distance(store_loc, customer_loc)))
            fee = int(Basic.distance(store_loc, customer_loc)*120) + 2500 #2500
        else:
            fee = fee
        #fee = add_fee
        #input('고객 {} 수수료 {}'.format(int(data[0]), fee))
        c = Basic.Customer(env, int(data[0]), input_location = [store_loc, customer_loc], fee= fee, end_time= customer_wait_time, far = int(data[6]))
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
                input("check")
        if env.now > end_time:
            break



def DriverMaker(env, driver_dict, customer_set ,speed = 2, end_time = 800, intervals = [], interval_para = False, interval_res = [],
                toCenter = True, error = 0, run_time = 900, pref_info = None, driver_left_time = 120, print_para = False,
                start_pos = [26,26], value_cal_type = 'return', num_gen = 10):
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
                            pref_info= pref_info, left_time=driver_left_time, print_para = print_para, start_pos= start_pos, value_cal_type = value_cal_type)
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
        #print('rider', rider.name, 'gen', env.now)
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
        for line in lines[2:]:
            interval = line.split(',')
            interval = interval[2:len(interval)]
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
