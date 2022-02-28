# -*- coding: utf-8 -*-
#.py 설명 :
# -> Basic_class.py : 기본 함수

import LinearizedASP_gurobi as lpg
import random
import simpy
import math
import operator
import copy
import numpy as np
from ValueRevise import RiderWeightUpdater


global shortest_path_data_np
shortest_path_data_np = np.load('rev_송파구_shortest_path_Distance_data0.npy')


class Customer(object):
    def __init__(self, env, name, input_location, end_time = 800, ready_time=2, service_time=1, fee = 1000,wait = True, far = 0, type_num = 4):
        """
        고객 class
        :param env: simpy Environment
        :param name: 이름 int+
        :param input_location: [주문한 가게 위치, 고객 위치] <-[[x1,y1],[x2,y2]]
        :param end_time: 주문 종료 시점<- 주문 발생 후 이 시간 동안 서비스를 받지 못하는 경우 주문 취소됨
        :param ready_time: 가게에서 준비시간(분) float+
        :param service_time: 고객에게 도차가 후 소요되는 서비스 시간(분) float+
        :param fee: 배송료
        :param wait: 고객을 일정간격 마다 생성하기 위한 장치.
        :param far: 고객의 종류 <-추후 실험을 위한 부분
        :param end_type: 고객의 종류 <-추후 실험을 위한 부분
        """
        self.name = name  # 각 고객에게 unique한 이름을 부여할 수 있어야 함. dict의 key와 같이
        self.time_info = [round(env.now, 2), None, None, None, None, end_time, ready_time, service_time]
        # [0 :발생시간, 1: 차량에 할당 시간, 2:차량에 실린 시간, 3:목적지 도착 시간,
        # 4:고객이 받은 시간, 5: 보장 배송 시간, 6:가게에서 준비시간,7: 고객에게 서비스 하는 시간]
        self.location = input_location
        self.assigned = False
        self.loaded = False
        self.done = False
        self.cancelled = False
        self.server_info = None
        self.fee = [fee, 0, None, None] # [기본 요금, 지급된 보조금, 할당된 라이더]
        self.wait = wait
        self.far = far
        self.error = 0
        self.type = random.choice(list(range(type_num))) #[0,1,2,3]#random.randrange(1,end_type) #random.randrange(1,end_type)
        env.process(self.Decline(env))

    def Decline(self, env, slack = 10):
        """
        고객이 생성된 후 자신의 종료 시점(end_time)+여유시간(default : 10) 동안 서비스를 받지 못하면 고객을 취소 시킴
        취소된 고객은 더이상 서비스 받을 수 없음. 단, 현재 서비스 받고 있는 경우에는 서비스를 받는 것으로 함.
        :param env:simpy Environment
        """
        yield env.timeout(self.time_info[5] + slack)
        if self.assigned == False and self.done == False:
            self.cancelled = True
            self.server_info= ["N", round(env.now,2)]


class Rider(object):
    def __init__(self, env, name, speed, customer_set, wageForHr = 9000, wait = True, toCenter = True, run_time = 900, error = 0,
                 ExpectedCustomerPreference = [0,250,500,750], pref_info = 'None', save_info = False, left_time = 120, print_para = False,
                 start_pos = [26,26], value_cal_type = 'return',coeff_revise_option = False, weight_sum = False, Data = None):
        """
        라이더 class
        :param env: simpy Environment
        :param name: 라이더 이름 int+
        :param speed: 라이더 속도(meter/분)
        :param customer_set: 고객 dict <- 생성된 고객이 저장되는 dict
        :param wageForHr: 라이더가 생각하는 시간당 임금(원)
        :param wait: 고객을 일정간격 마다 생성하기 위한 장치.
        :param toCenter: PriorityOrdering에서 라이더의 가치를 판단하는 방식(toCenter == True시 라이더가 다시 center로 돌아가는 것 까지 운행 비용에 고려함)
        :param run_time: 시뮬레이션 러닝 시간(분)
        :param error: 라이더에 대해서 플랫폼이 가지는 error
        :param ExpectedCustomerPreference: 고객 선호도<-추후 실험을 위한 장치
        :param pref_info: PriorityOrdering에서 가치 함수 계산에 사용되는 방법
        :param save_info: PriorityOrdering에서 주문 정보를 추가적으로 저장할 것인지 선택. True / False
        :param left_time: 라이더가 발생 후 근무하는 시간
        :param print_para: print문 실행 파라메터
        :param start_pos: 라이더 시작 위치 [x1,y1]
        :param value_cal_type:고객의 가치를 계산하는 방식
        """
        self.env = env
        self.name = name
        self.veh = simpy.Resource(env, capacity=1)
        self.last_location = start_pos # [26, 26]  # 송파 집중국
        self.served = []
        self.speed = speed
        self.wageForHr = wageForHr
        self.idle_times = [[],[]]
        self.gen_time = int(env.now)
        self.left_time = None
        self.wait = wait
        self.end_time = env.now
        self.left = False
        self.earn_fee = []
        self.fee_analyze = []
        self.subsidy_analyze = []
        self.choice = []
        self.choice_info = []
        self.now_ct = [1,1]#[[26,26],[26,26]]
        self.w_data = []
        for slot_num in range(int(math.ceil(run_time / 60))):
            self.fee_analyze.append([])
            self.subsidy_analyze.append([])
        self.exp_last_location = start_pos
        self.error = int(error)
        pref = list(range(1, len(ExpectedCustomerPreference) + 1))
        random.shuffle(pref)
        self.CustomerPreference = pref
        self.expect = ExpectedCustomerPreference
        cost_coeff = round(random.uniform(0.2,0.45),2)#round(random.uniform(0.3,1.0),1)
        type_coeff = 0.6 - cost_coeff#3 - (1.5 + cost_coeff) #round(random.uniform(0.8,1.2),1)
        self.coeff = [cost_coeff,type_coeff,0.4]#[cost_coeff,type_coeff,1.5] #[cost_coeff,type_coeff,1] #[1,1,1]
        self.p_coeff = [1,1,1] #[0.9,0.4,0.8] #[1,1,1]#[거리, 타입, 수수료]
        self.past_route = []
        self.past_route_info = []
        self.P_choice_info = []
        self.p_history = [copy.deepcopy(self.p_coeff)]
        self.violated_choice_info = []
        self.LP1History = []
        self.LP2History = []
        self.LP3History = []
        self.LP3_2History = []
        self.LP1p_coeff = [0.3,0.3,0.3] #[1,1,1]
        self.LP2p_coeff = [0.3,0.3,0.3] #[1,1,1]
        self.LP3p_coeff = [0.3,0.3,0.3] #[1,1,1]
        self.LP3_2p_coeff = [0.3,0.3,0.3]
        self.validations = [0,0,0,0]
        self.validations_detail = [[],[],[],[]]
        self.validations_detail_abs = [[], [], [], []]
        self.weight_update_function = False
        env.process(self.Runner(env, customer_set, toCenter = toCenter, pref = pref_info, save_info = save_info,
                                print_para = print_para,coeff_revise_option = coeff_revise_option, weight_sum= weight_sum, Data= Data))
        env.process(self.RiderLeft(left_time))


    def RiderLeft(self, working_time):
        """
        일을 시작한 working_time동안 운행 후, 기사는 시장에서 이탈.
        :param left_time: 운행 시간.
        """
        yield self.env.timeout(working_time)
        self.left = True
        self.left_time = int(self.env.now)
        print('라이더 {} 운행 종료 T {}'.format(self.name, self.env.now))


    def CustomerSelector(self, customer_set, toCenter = True, pref = 'None', save_info = False, value_cal_type = 'return', print_para = False):
        """
        고객 (customer_set)에 대해 PriorityOrdering를 통해, 고객들의 가치를 계산.
        계산 된 고객들 중 가치가 가장 큰 고객을 선택.
        :param customer_set: 고객 dict [class customer, class customer,...,]
        :param toCenter: PriorityOrdering에서 라이더의 가치를 판단하는 방식(toCenter == True시 라이더가 다시 center로 돌아가는 것 까지 운행 비용에 고려함)
        :param pref: PriorityOrdering에서 가치 함수 계산에 사용되는 방법
        :param save_info: PriorityOrdering에서 주문 정보를 추가적으로 저장할 것인지 선택. True / False
        :param value_cal_type : 고객의 가치를 계산하는 방식
        :return: 가장 높은 가치 고객 이름, PriorityOrdering 반환 값.
        """
        now_time = round(self.env.now,1)
        ava_cts = UnloadedCustomer(customer_set, now_time)
        ava_cts_class = []
        print('T:{}/ 라이더 {} 빈 고객 수{}'.format(int(self.env.now),self.name, len(ava_cts)))
        ava_cts_names = []
        if len(ava_cts) > 0:
            if type(ava_cts[0]) == int:
                for ct_name in ava_cts:
                    ava_cts_class.append(customer_set[ct_name])
                ava_cts_names = ava_cts
            else:
                ava_cts_class = ava_cts
                for info in ava_cts:
                    ava_cts_names.append(info.name)
        if len(ava_cts_class) > 0:
            print('라이더 선택 시점의 라이더 위치{}'.format(self.last_location))
            #print('고를 수 있는 고객 들 수',len(ava_cts_class))
            print('T {} 라이더 선택 시점의 설정 {} {} {}'.format(self.env.now, toCenter, pref, value_cal_type))
            priority_orders = PriorityOrdering(self, ava_cts_class, toCenter = toCenter, who = pref, save_info = save_info, rider_route_cal_type = value_cal_type)
            priority_orders_biggerthan1 = []
            for info in priority_orders:
                if info[1] > 0:
                    priority_orders_biggerthan1.append(info)
            try:
                print('1등 주문',priority_orders_biggerthan1)
            except:
                pass
            #print(priority_orders)
            #print('이득이 되는 고객 들 수', len(priority_orders_biggerthan1))
            #print('rider', self.name,'//Now',round(self.env.now,2),'//un_ct',len(ava_cts),'//candidates', priority_orders[:min(3,len(priority_orders))],'//ava_cts:',ava_cts_names)
            #input('Stop')
            if print_para == True:
                print('라이더::{} 위치 {}::이윤 계산결과{}'.format(self.name, self.last_location, priority_orders))
                print("라이더::",self.name,"/이윤 정보::",priority_orders_biggerthan1)
            if len(priority_orders_biggerthan1) > 0:
                ct = customer_set[priority_orders_biggerthan1[0][0]]
                print(self.name, 'selects', ct.name, 'at', self.env.now)
                self.w_data.append([int(self.env.now),ct.name, ava_cts_names])
                return ct.name, priority_orders_biggerthan1
            else:
                return None, None
        return None, None


    def Runner(self, env, customer_set, wait_time=1, toCenter = True, pref = 'None', save_info = False, print_para = False,
               coeff_revise_option = False, weight_sum = False, Data = None):
        """
        라이더의 운행을 표현.
        if 수행할 주문이 있는 경우:
        ---주문을 수행
        else:
        ---대기
        :param env:simpy Environment
        :param customer_set: 고객 dict [class customer, class customer,...,]
        :param wait_time: 현재 수행하고 있는 주문이 없는 경우, 다음 주문 선택까지의 대기 시간.
        :param toCenter: PriorityOrdering에서 라이더의 가치를 판단하는 방식(toCenter == True시 라이더가 다시 center로 돌아가는 것 까지 운행 비용에 고려함)
        :param pref: PriorityOrdering에서 가치 함수 계산에 사용되는 방법
        :param save_info: PriorityOrdering에서 주문 정보를 추가적으로 저장할 것인지 선택. True / False
        """
        while self.left == False:
            #print('라이더::', self.name, '시간::',env.now)
            if len(self.veh.put_queue) == 0 and self.wait == False:
                #print('Rider', self.name, 'assign1 at', env.now)
                ct_name, infos = self.CustomerSelector(customer_set, toCenter = toCenter, pref = pref, save_info = save_info, print_para = True)
                if infos != None: #infos == None인 경우에는 고를 고객이 없다는 의미임.
                    """
                    rev_infos = []
                    for info in infos:
                        rev_infos.append([info[0],info[2]])
                    """
                    #print('T:{}/ 라이더 {} 고객 없음'.format(int(env.now), self.name))
                    if pref == 'test_rider' or pref == 'test_platform':
                        #self.choice_info.append([int(env.now), ct_name, self.last_location , rev_infos])
                        self.choice_info.append([int(env.now), ct_name, self.last_location, infos])
                        if self.weight_update_function == True:
                            print('갱신 확인1 라이더 {}/LP1{} LP2 {} LP3{}'.format(self.name, self.LP1p_coeff, self.LP2p_coeff,
                                                                            self.LP3p_coeff))
                            RiderWeightUpdater(self, customer_set, weight_sum = True, beta=1) #todo 220225 : 라이더가 선택후 각 방식에 의해 rider weighr 갱신 수행
                            print('라이더 가중치 {}'.format(self.coeff))
                            print('갱신 확인2 라이더 {}/LP1{} LP2 {} LP3{}'.format(self.name, self.LP1p_coeff, self.LP2p_coeff, self.LP3p_coeff))
                        #print('선택 정보 저장 {}'.format())
                #print('Rider',self.name,'assign2',ct_name, 'at', env.now)
                if infos == None:
                    pass
                else:
                    if print_para == True:
                        print('Now',int(env.now),'Rider ::',self.name ,' /select::', infos[:min(3,len(infos))])
                    else:
                        pass
                select_time = round(env.now,2)
                if type(ct_name) == int and ct_name > 0:
                    ##라이더 가치함수 갱신이 여기서 수행되어야 함.
                    # 현재 이 라이더가 보고 있는 주문들에 대해서 플랫폼의 문제를 풀고, 이에 대한 답이 다른 경우에 갱신할 필요가 있음.
                    if coeff_revise_option == True:
                        P_ct_name, P_infos = self.CustomerSelector(customer_set, toCenter=toCenter, pref= 'test_platform',
                                                               save_info=save_info, print_para=True)
                        if P_ct_name != ct_name:
                            others = []
                            for info in infos:
                                others.append([-info[2], -info[7], info[8]])
                            selected = copy.deepcopy(others[0])
                            #input('갱신 문제 시도 {}'.format(others))
                            if len(others) > 1:
                                print('갱신 문제 시작 {}:과거 데이터{}'.format(others,self.P_choice_info))
                                coeff_update_feasibility, res = lpg.ReviseCoeffAP2(selected, others[1:], self.p_coeff, past_data = self.P_choice_info, weight_sum = weight_sum)
                                self.P_choice_info.append([others[0], others[1:]])
                                # 계수 갱신
                                if coeff_update_feasibility == True:
                                    print("목표::", self.coeff)
                                    print("갱신전::", self.p_coeff)
                                    for index in range(len(res)):
                                        self.p_coeff[index] += res[index]
                                    print('갱신됨::{}'.format(self.p_coeff))
                    self.choice.append([ct_name, round(env.now,4)])
                    ct = customer_set[ct_name]
                    self.now_ct = ct.location
                    self.earn_fee.append(ct.fee[1])
                    if ct.fee[1] > 0:
                        #input('고객{} 보조금{}'.format(ct.name, ct.fee))
                        pass
                    ct.assigned = True
                    ct.time_info[1] = round(env.now, 2)
                    #input('고객 이름{}; 고객 {} ;보조금 정보 {}; 시간 정보{}'.format(self.now_ct, ct.name, ct.fee, ct.time_info))
                    end_time = env.now + (distance(ct.location[0],self.last_location) / self.speed) + ct.time_info[6] #self.last_location == 고객, ct.location[0] = 가게
                    end_time += ((distance(ct.location[0], ct.location[1]) / self.speed) + ct.time_info[7])
                    if int(env.now // 60) >= len(self.fee_analyze):
                        print(env.now, self.fee_analyze)
                    try:
                        self.fee_analyze[int(env.now // 60)].append(ct.fee[0])
                        self.subsidy_analyze[int(env.now // 60)].append(ct.fee[1])
                    except:
                        input('fee_analyze 고장')
                        pass
                    self.end_time = end_time
                    print('라이더{}종료시간:{}'.format(self.name, self.end_time))
                    self.exp_last_location = ct.location[1]
                    #print('Rider', self.name, 'select', ct_name, 'at', env.now, 'EXP T', self.end_time)
                    #print('1:', self.last_location, '2:', ct.location)
                    with self.veh.request() as req:
                        print('라이더 과거 경로',self.past_route[-2:],'고객 경로', ct.location)
                        print('라이더 과거 경로', self.past_route_info[-2:], '고객 경로', ct.location)
                        #print(self.name, 'select', ct.name, 'Time:', env.now)
                        req.info = [ct.name, round(env.now,2)]
                        yield req  # users에 들어간 이후에 작동
                        time = distance(ct.location[0],self.last_location) / self.speed
                        #print('With in 1:',self.last_location, '2:', ct.location[0])
                        time += ct.time_info[6]
                        #end_time += time
                        ct.loaded = True
                        #ct.time_info[2] = round(env.now, 2)
                        yield env.timeout(time)
                        ct.time_info[2] = round(env.now, 2)
                        time = distance(ct.location[0], ct.location[1]) / self.speed
                        time += ct.time_info[7]
                        #end_time += time
                        self.served.append([ct.name, 0])
                        self.last_location = ct.location[0]
                        self.past_route.append(ct.location[0])
                        self.past_route_info.append([ct.name,0,ct.location[0]])
                        #print('3:', ct.location[1])
                        yield env.timeout(time)
                        ct.time_info[3] = round(env.now, 2) - ct.time_info[7]
                        ct.time_info[4] = round(env.now,2)
                        ct.done = True
                        ct.server_info = [self.name, round(env.now,2)]
                        self.served.append([ct.name,1])
                        self.last_location = ct.location[1]
                        self.past_route.append(ct.location[1])
                        self.past_route_info.append([ct.name, 1, ct.location[1]])
                        #임금 분석
                        print('Rider', self.name, 'done', ct_name, 'at', int(env.now))
                else:
                    self.end_time = env.now + wait_time
                    self.idle_times[0].append(wait_time)  #수행할 주문이 없는 경우
                    yield self.env.timeout(wait_time)
                    #print('Rider', self.name, '유효 주문X at', int(env.now),"/이득이 되는 주문 수", type(infos))
                    if type(infos) == list:
                        print("이득이 되는 주문 수",len(infos))
            else:
                self.end_time = env.now + wait_time
                self.idle_times[1].append(wait_time) #이미 수행하는 주문이 있는 경우
                yield self.env.timeout(wait_time)
                print('T:{}/ 라이더 {} 주문 없음으로 대기'.format(int(env.now), self.name))


def UnloadedCustomer(customer_set, now_time):
    """
    고객들(customer_set) 중 아직 라이더에게 할당되지 않은 고객들 계산
    :param customer_set: 고객 dict [class customer, class customer,...,]
    :param now_time: 현재시간
    :return: [고객 class, ...,]
    """
    res = []
    for ct_name in customer_set:
        customer = customer_set[ct_name]
        cond1 = now_time - customer.time_info[0] < customer.time_info[5]
        cond2 = customer.assigned == False and customer.loaded == False and customer.done == False
        cond3 = customer.wait == False
        #print('CT check',cond1, cond2, cond3, customer.name, customer.time_info[0])
        #input('STOP')
        if cond1 == True and cond2 == True and cond3 == True and ct_name > 0 and customer.server_info == None:
            res.append(customer)
    return res

def CheckTimeFeasiblity(veh, customer, customers, toCenter = True, rider_route_cal_type = 'return', last_location = None, who = 'driver',init_center = 1):
    """
    입력 받은 차량(veh)가 고객(customer)를 제한 시간 내에 방문할 수 있는지 여부(T/F)와 그 비용을 계산
    :param veh: class Rider
    :param customer: 해당 고객 class Customer
    :param customers: 전체 고객 dict -> 라이더 가치 계산을 위해 사용됨.
    :param toCenter: PriorityOrdering에서 라이더의 가치를 판단하는 방식(toCenter == True시 라이더가 다시 center로 돌아가는 것 까지 운행 비용에 고려함)
    :param last_location: 차량의 현재 위치를 설정. None:차량의 현재 위치 / [x,y]: 이 지점을 차량의 시작 위치로 설정.
    :return:
    """
    now_time = round(veh.env.now, 1)
    if rider_route_cal_type == 'return':
        if who == 'test_platform':
            rev_last_location = veh.now_ct[1]
            time = CalTime2(rev_last_location, veh.speed, customer, center=init_center, toCenter=toCenter,customer_set=customers)
            #print('플랫폼 시점의 라이더 위치{}:: 고객이름{} ::복귀{} ::시간{}'.format(rev_last_location, customer.name,  [25,25],time))
        else:
            time = CalTime2(veh.last_location, veh.speed, customer, center=init_center, toCenter=toCenter,customer_set=customers)
            #print('라이더 시점의 라이더 위치{}:: 고객이름{} ::복귀{} ::시간{}'.format(veh.last_location,customer.name,  [25,25],time))
    elif rider_route_cal_type == 'no_return':
        time = CalTime2(veh.last_location, veh.speed, customer, center=customer.location[1], toCenter=toCenter,
                        customer_set=customers)
    elif rider_route_cal_type == 'nearest_order_store':
        time = CalTime2(veh.last_location, veh.speed, customer, center=customers[0].location[0], toCenter = False,
                        customer_set=customers)
    elif rider_route_cal_type == 'next_order':
        time = CalTime2(last_location, veh.speed, customer, center=customer.location[1], toCenter=toCenter,
                        customer_set=customers)
    else:
        pass
    cost = (time / 60) * veh.wageForHr
    ###이 고객이 자신의 end_time 내에 서비스 받을 수 있는지 계산.###
    t2 = time - distance(customers[0].location[0],customer.location[1]) / veh.speed
    time_para = now_time + t2 < customer.time_info[0] + customer.time_info[5]  # time_para == True인 경우는 해당 고객이 자신의 end_time이전에 서비스 받을 수 있음을 의미.
    #print('고객 {} 기준 비용 {}'.format(customer.name , cost))
    return time_para, cost, round(time,1)

def PriorityOrdering(veh, customers, minus_para = False, toCenter = True, who = 'test_rider', save_info = False, sort_para = True, rider_route_cal_type = 'return', last_location = None, LP_type = 'LP2'):
    """
    veh의 입장에서 customers의 고객들을 가치가 높은 순서대로 정렬한 값을 반환
    :param veh: class rider
    :param customers: now_time에 라이더가 선택할 수 있는 주문들(unassigned customers)
    :param now_time: 현재 시간.
    :param minus_para: 주문의 가치가 음수인 경우에 0을 반환
    :param toCenter: PriorityOrdering에서 라이더의 가치를 판단하는 방식(toCenter == True시 라이더가 다시 center로 돌아가는 것 까지 운행 비용에 고려함)
    :param who: 누구의 입장에서 고객 가치를 계산하는지 설정. driver/ test_rider / test_platform
    :param save_info: PriorityOrdering에서 주문 정보를 추가적으로 저장할 것인지 선택. True / False
    :param last_location: 차량의 현재 위치를 설정. None:차량의 현재 위치 / [x,y]: 이 지점을 차량의 시작 위치로 설정.
    :return:[[고객 이름, 이윤],...] -> 이윤에 대한 내림차순으로 정렬됨.
    """
    res = []
    incentive_list = []
    for customer in customers:
        time_para, cost, time = CheckTimeFeasiblity(veh, customer, customers, toCenter = toCenter, rider_route_cal_type = rider_route_cal_type, last_location = last_location, who = who)
        fee = customer.fee[0]
        org_cost = copy.deepcopy(cost)
        paid = 0
        save_fee = customer.fee[0]
        if customer.fee[2] == veh.name or customer.fee[2] == 'all':
            fee += customer.fee[1]
            paid += customer.fee[1]
            save_fee += customer.fee[1]
            if customer.fee[1] > 0:
                incentive_list.append([customer.name, customer.fee[1]])
        cost2 = 0
        if who == 'platform':
            cost2 = veh.error
        elif who == 'test_rider':
            fee = fee* veh.coeff[2]
            #print(veh.expect, customer.type,veh.coeff[1])
            #input('확인')
            cost2 = veh.expect[customer.type] * veh.coeff[1]
            cost = cost*veh.coeff[0]
        elif who == 'test_platform':
            if LP_type == 'LP1':
                fee = fee * veh.LP1p_coeff[2]
                cost2 = veh.expect[customer.type] * veh.LP1p_coeff[1]
                cost = cost*veh.LP1p_coeff[0]
            elif LP_type == 'LP2':
                fee = fee * veh.LP2p_coeff[2]
                cost2 = veh.expect[customer.type] * veh.LP2p_coeff[1]
                cost = cost*veh.LP2p_coeff[0]
            elif LP_type == 'LP3':
                fee = fee * veh.LP3p_coeff[2]
                cost2 = veh.expect[customer.type] * veh.LP3p_coeff[1]
                cost = cost*veh.LP3p_coeff[0]
            else:
                fee = fee * veh.p_coeff[2]
                cost2 = veh.expect[customer.type] * veh.p_coeff[1]
                cost = cost*veh.p_coeff[0]

        else:
            pass
        #print('cost2', cost2,'::', fee - cost- cost2)
        end_slack_time = veh.env.now - (customer.time_info[0] + customer.time_info[5] + 10)
        if time_para == True:
            if minus_para == True:
                res.append([customer.name, max(0,int(fee - cost - cost2)), int(org_cost), int(fee), time, 'Profit1',cost, customer.type,save_fee,end_slack_time],)
            elif fee > cost + cost2 :
                res.append([customer.name, int(fee - cost - cost2), int(org_cost), int(fee), time, 'Profit2',cost, customer.type,save_fee,end_slack_time])
            else:
                #print('negative value',int(fee - cost- cost2))
                res.append([customer.name, int(fee - cost - cost2), int(org_cost), int(fee), time,'N/A1',cost, customer.type,save_fee,end_slack_time])
                pass
        else:
            res.append([customer.name,int(fee - cost - cost2),int(org_cost),int(fee), time,'N/A2',cost, customer.type, save_fee,end_slack_time])
        res[-1].append(cost2)
        res[-1].append(customer.type)
    if len(res) > 0:
        if sort_para == True:
            res.sort(key=operator.itemgetter(1), reverse = True)
    if len(incentive_list) > 0:
        print('선택 정보 {} '.format(res))
        #input('라이더 선택시 인센 정보 {}'.format(incentive_list))
    return res


def CalTime2(veh_location,veh_speed, customer, center=1, toCenter = True, customer_set = []):
    """
    cost(1) : customer를 서비스하는 비용
    cost(2) : 종료 후 다시 중심으로 돌아오는데 걸리는 시간.
    :param veh_location: 차량의 시작 위치
    :param veh_speed: 차량 속도
    :param customer: 고객
    :param center: 중심지의 위치(가게들이 밀접한 지역)
    :return: 필요한 시간
    """
    #print('Cal Time2',veh_location, customer.location, center)
    time = distance(customer.location[0],veh_location) / veh_speed #자신의 위치에서 가게 까지
    #time += distance(customer.location[0], customer.location[1]) / veh_speed
    t2 = distance(customer.location[0], customer.location[1]) / veh_speed # 가게에서 고객 까지
    #print('t1:',int(time), 't2:', int(t2))
    time += t2
    time += (customer.time_info[6] + customer.time_info[7]) #고객 서비스 시간.
    if toCenter == True:
        #print('거꾸로?')
        time += distance(center,customer.location[1])/veh_speed
    else: #라이더가 돌아가는 위치가 현재 가능한 주문들 중 가장 가까운 가게라고 계산하는 경우
        dist = []
        for ct in customer_set:
            dist.append([ct.name, distance(ct.location[0],customer.location[1])])
        if len(dist) > 0:
            dist.sort(key=operator.itemgetter(1))
            time += dist[0][1]/ veh_speed
            #aveage = []
            #for info in dist:
            #    aveage.append(info[1])
            #time += sum(aveage)/len(aveage)
    return time


def InitializeSubsidy(customer_set):
    """
    customer_set의 보조금을 모두 초기화 한다.
    :param customer_set: 고객 dict [class customer, class customer,...,]
    :return:
    """
    for ct_name in customer_set:
        ct = customer_set[ct_name]
        if ct.assigned == False or ct.time_info == None:
            ct.fee[1] = 0
            ct.fee[2] = None
    return None

def DefreezeAgent(object_dict, type = 'rider'):
    """
    object_dict에 있는 고객들을 subsidy문제가 고려할 수 있는 상태로 변환
    :param object_dict: object dict customer or rider [class, class,...,]
    :param type: Defreeze하는 대상 rider / customer
    :return:
    """
    if type == 'rider':
        for object_name in object_dict:  # interval이 끝난 후 새롭게 들어온 라이더와 고객을 시스템에 반영시킴
            rider = object_dict[object_name]
            if rider.left == False and rider.wait == True:
                rider.wait = False
    elif type == 'customer': #type == 'customer'
        for object_name in object_dict:
            customer = object_dict[object_name]
            if customer.wait == True and customer.name > 0:
                customer.wait = False
    else:
        input("Agent name error")
    return None

def AvaRider(rider_set, now_time,interval = 10):
    """
    (현재시점(now_time)~ 현재 시점 + interval) 사이에 주문을 선택할 수 있는 라이더를 계산
    :param rider_set: 라이더 dict [class rider, class rider, ...,]
    :param now_time: 현재 시점
    :param interval: 시간 간격
    :return:
    """
    res = []
    for rider_name in rider_set:
        rider = rider_set[rider_name]
        cond1 = rider.wait == False
        cond2 = now_time < rider.end_time < now_time + interval
        cond3 = rider.left == False
        #print(rider.name, '::', cond1, '::',cond2, '::',cond3)
        if cond1 == True and cond2 == True and cond3 == True:
            res.append(rider.name)
            print(rider.name,'::',rider.end_time ,'<', now_time + interval)
    print('AvaRider#', len(res))
    return res


def WhoGetPriority(customers , cut, now_time, time_thres = 0.8, print_para = False, speed = 1.5):
    """
    종료시점이 임박한 고객들에 대하여, 종료시점이 임박한 고객을  cut만큼 반환
    :param customers: [고객 class,...,]
    :param ava_riders_num:가능한 라이더의 수(이 수만큼만 고객들을 뽑기 때문)
    :param now_time: 현재 시간
    :param time_thres: 고객의 endtime 중 얼마의 시간이 지난 경우 우선 고객으로 지정하는가? (0<x<1)
    :return: [고객 이름1, 고객 이름2,...ㅡ]
    """
    scores = []
    index = 0
    print('candidates',len(customers),'ava_riders_num', cut)
    test = []
    for customer in customers:
        test.append(int(now_time - customer.time_info[0]))
        required_time = distance(customer.location[0], customer.location[1])/speed + customer.time_info[6] + customer.time_info[7]
        if (customer.time_info[0] + customer.time_info[5] - now_time)*time_thres <= required_time :
        #if now_time - customer.time_info[0] > customer.time_info[5]*time_thres:
            #scores.append([customer.name, customer.time_info[0], index])
            scores.append([customer.name, now_time -(customer.time_info[0]+required_time), index])
        index += 1
    if print_para == True:
        print('urgent ct info',scores)
    #print('t test', test)
    if len(scores) > 0:
        scores.sort(key=operator.itemgetter(1))
        res = []
        tem = []
        for info in scores[:cut]:
            res.append(info[2])
            tem.append(info[0])
        return res, tem
    return [], []


def distance(p1, p2):
    """
    두 지점 사이의 거리를 반환
    :param p1: [x,y]
    :param p2: [x,y]
    :return: 거리
    """
    global shortest_path_data_np
    if type(p1) == list and type(p2) == list:
        res = round(math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), 2)
    else:
        if type(p1) != int or type(p2) != int:
            print(p1, p2)
            input('ERROR2')
        store_max_index = np.shape(shortest_path_data_np)[0]
        customer_max_index = np.shape(shortest_path_data_np)[1]
        if p1 > store_max_index or p2 > customer_max_index:
            print('거꾸로 필요',p1,'=<', store_max_index, '::',p2,'=<', customer_max_index)
            #input('error')
        try:
            res = float(shortest_path_data_np[p1,p2]*1000)
        except:
            res = float(shortest_path_data_np[p2, p1] * 1000)
    return res


def NextMin(lamda):
    """
    시간 당 lamda에 대해 다음 사건 발생 간격(분)을 반환
    :param lamda : lamda
    :return next_min: 다음 발생 간격
    """
    # lambda should be float type.
    # lambda = input rate per unit time
    next_min = (-math.log(1.0 - random.random()) / lamda)*60
    return float("%.2f" % next_min)

