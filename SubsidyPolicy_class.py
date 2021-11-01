# -*- coding: utf-8 -*-
#.py 설명 :
# -> SubsidyPolicy_class.py : 보조금 문제 풀이 관련 함수

import operator
import numpy as np
import LinearizedASP_gurobi as lpg
import Basic_class as Basic

def indexReturner2DBiggerThan0(list2D, val = 0):
    """
    2차원 행렬인 list2D에서 val보다 더 큰 값을 가지는 값을 반환
    :param list2D: 값이 저장된 행렬
    :param val: 비교 값. defauly = 0
    :return:list2D에서 val 보다 더 큰 값 array [[row index,column index, 값],...]
    """
    res = []
    for row in range(len(list2D)):
        for index in range(len(list2D[row])):
            if int(list2D[row][index]) > val:
                res.append([row, index, list2D[row][index]])
    return res


def FeeUpdater(rev_v, customer_set, riders, rider_set ,cts_name, now_time, subsidy_offer = [],subsidy_offer_count= [], upper = 10000, print_para = False):
    """
    계산한 보조금을 고객들에게 반영함.
    :param rev_v: 계산된 보조금 결과
    :param customer_set: 고객 dict [class customer, class customer,...,]
    :param riders: 주문을 수행할 수 있는 라이더들의 이름 list [name, name,...,]
    :param rider_set: 라이더 dict [class rider, class rider, ...,]
    :param cts_name: 가능한 주문들의 이름 list [name, name,...,]
    :param now_time: 현재 시점
    :return: True
    """
    for info in rev_v:
        try:
            rider_name = riders[info[0]]
            rider = rider_set[rider_name]
            customer = customer_set[cts_name[info[1]]]
            if info[2] < upper and customer.name > 0:
                customer.fee[1] = info[2] + 10
                customer.fee[2] = rider.name
                customer.fee[3] = now_time
                subsidy_offer.append([rider.name, customer.name])
                if print_para == True:
                    print('Fee offer to rider#',rider.name,'//end T:', round(rider.end_time,2),'//Rider left t',round(rider.gen_time + 120,2),'//CT #',customer.name,'//CT end T', customer.time_info[0] + customer.time_info[5],'$',customer.fee[0] + customer.fee[1], '//offered$', customer.fee[1])
                subsidy_offer_count[int(now_time//60)] += 1
            else:
                pass
        except:
            print('Dummy customer')
            pass
    return True


def solver_interpreterForIP2(res_v):
    """
    Gurobi solver가 푼 IP 문제의 해의 feasilbility를 출력하고, 제안된 보조금(v)를 반환
    :param res_v: IP 문제를 푼 get.Vars()
    :return: feasibility, v
    """
    if type(res_v) == list:
        v = [[0,0,res_v[0]]]
    else:
        v = indexReturner2DBiggerThan0(res_v, 1)
        # [행, 렬, 할당 된 보조금 값]
    #print('check v',v)
    if len(v) > 0:
        return True, v
    else:
        return False, None


def CalculateRiderOrder(rider_set, now_time):
    """
    주어진 rider set에 대해서, 라이더의 선택 시점 순서 반환
    :param rider_set: 라이더 dict [class rider, class rider, ...,]
    :param now_time: 현재 시점
    :return: 라이더들의 순서
    """
    d_orders = []
    riders = Basic.AvaRider(rider_set, now_time)
    for rider_name in riders:
        rider = rider_set[rider_name]
        d_orders.append([rider_name, rider.end_time])
    #d_orders를 계산
    d_orders.sort(key = operator.itemgetter(1))
    d_orders_res = [] #todo : 현재는 라우트의 길이가 짧을 수록 고객을 더 먼저 선택할 것이라고 가정함.
    for rider_name in riders:
        index = 1
        for info in d_orders:
            if rider_name == info[0]:
                d_orders_res.append(index)
                break
            index += 1
    #print('d_orders_res',d_orders_res)
    return d_orders_res


def InputValueReform(v_old, times, end_times, customer_num,rider_num,upper, dummy_customer_para = True):
    """
    v_old, times, end_times에 대한 reform을 수행.
    (1) rider_num > customer_num :  |rider_num - customer_num| 수의 더미 고객 생성
    (2) dummy_customer_para == True : rider수에 대한 더미 고객 생성 ->rider가 주문을 선택하지 않을 수 있는 경우 표현
    :param v_old: 플랫폼이 예상한, 라이더가 고객들에 대해 가지는 가치
    :param times: 주문 완료 시점
    :param end_times: 주문 종료 시점(고객이 기다리다 떠나는 시점)
    :param customer_num: 고객 수 int+
    :param rider_num: 라이더 수 int+
    :param upper: 보조금 상한
    :param dummy_customer_para: (2)를 작동 시키는지 여부 결정 T/F
    :return: v_old, times, end_times
    """
    if rider_num > customer_num:
        dif = rider_num - customer_num
        add_array1 = np.zeros((rider_num,dif))
        v_old = np.hstack((v_old, add_array1))
        add_array2 = np.zeros((rider_num,dif))
        add_array3 = np.zeros((rider_num,dif))
        for i in range(0,len(add_array2)):
            for j in range(0,len(add_array2[i])):
                add_array2[i,j] = 1000
        times = np.hstack((times, add_array3))
        end_times = np.hstack((end_times, add_array2))
    if rider_num > 1 and dummy_customer_para == True:
        add_array4 = np.zeros((rider_num, rider_num))
        #print("add_array4사이즈",np.sign(add_array4))
        for index in range(0,len(add_array4[0])):
            for row in range(0,len(add_array4)):
                add_array4[row, index] = -upper
        v_old = np.hstack((v_old, add_array4))
        add_array5 = np.zeros((rider_num, rider_num))
        add_array6 = np.zeros((rider_num, rider_num))
        for i in range(0,len(add_array5)):
            for j in range(0,len(add_array5[i])):
                add_array5[i,j] = 1000
        times = np.hstack((times, add_array6))
        end_times = np.hstack((end_times, add_array5))
    return v_old, times, end_times



def ProblemInput(rider_set, customer_set, now_time, minus_para = True, dummy_customer_para = True, upper = 1500, toCenter = True):
    """
    Monetary Incentive Allocation(SubsidyAllocation)문제를 풀기 위한 input 값을 계산
    :param rider_set:라이더 dict <- 생성된 라이더가 저장되는 dict
    :param customer_set: 고객 dict [class customer, class customer,...,]
    :param now_time: 현재 시점
    :param minus_para: True: 계산된 값이 음수인 경우 음수 대신 0을 value로 할당/ False: 계산된 값을 그대로 value에 할당
    :param dummy_customer_para: SubsidyAllocation 문제 정의 시, 라이더가 고객을 선택하지 않는 경우를 표현하는 파라메터. (if True : 더미 고객 정의 /if False: 더미 고객 정의 X)
    :param upper: 보조금 상한
    :param toCenter:PriorityOrdering에서 라이더의 가치를 판단하는 방식(toCenter == True시 라이더가 다시 center로 돌아가는 것 까지 운행 비용에 고려함)
    :return: v_old <- v_ij , riders <- 주문을 수행할 수 있는 라이더 , cts_name <- 주문을 수행 가능한 고객 이름,d_orders_res<- 라이더 주문 선택 순서
    times <- (라이더-주문) 에 대한 주문 완료에 걸리는 시간(분), end_times <- (주문)이 종료되는 시점
    """
    #print('Problem input')
    riders = Basic.AvaRider(rider_set, now_time)
    cts = Basic.UnloadedCustomer(customer_set, now_time)
    values = []
    d_orders = []
    times = []
    end_times = []
    ###v_old, times, end_times 계산###
    for rider_name in riders:
        rider = rider_set[rider_name]
        d_orders.append([rider_name, rider.end_time])
        ct_infos = Basic.PriorityOrdering(rider, cts, toCenter=toCenter, minus_para = minus_para, sort_para = False)
        for info in ct_infos:
            customer = customer_set[info[0]]
            values.append(info[1])
            end_times.append(customer.time_info[0] + customer.time_info[5])
            times.append(now_time + info[4])
    ###numpy 형식으로 재 저의###
    v_old = np.array(values).reshape(len(riders), len(cts))
    times = np.array(times).reshape(len(riders), len(cts))
    end_times = np.array(end_times).reshape(len(riders), len(cts))
    ###Input reform###
    #v_old, times, end_times = InputValueReform(v_old, times, end_times, len(cts), len(riders), upper, add_para = add)
    v_old, times, end_times = InputValueReform(v_old, times, end_times, len(cts), len(riders), upper, dummy_customer_para = dummy_customer_para)
    ###고객 이름 계산###
    cts_name = []
    for ct in cts:
        cts_name.append(ct.name)
    if len(riders) > len(cts): #더미 고객들 계산
        dif = len(end_times[0])-len(cts)
        for i in range(0,dif):
            cts_name.append(0)
    ###d_orders를 계산###
    d_orders_res = CalculateRiderOrder(rider_set, now_time)
    #print('d_orders_res',d_orders_res)
    return v_old, riders, cts_name, d_orders_res, times, end_times

def ExpectedSCustomer(rider_set, rider_names, d_orders_res, customer_set, now_time, toCenter = True, who = 'platform', print_para = False, rider_route_cal_type = 'return'):
    """
    d_orders_res 순서로 rider가 주문을 선택한다고 하였을 때, 선택될 고객들을 계산
    :param rider_set: 라이더 dict
    :param rider_names: 주문을 수행할 수 있는 라이더 이름
    :param d_orders_res: 라이더 주문 수행 순서 rider_names의
    :param customer_set: 고객 dict [class customer, class customer,...,]
    :param now_time: 현재 시점
    :param toCenter: PriorityOrdering에서 라이더의 가치를 판단하는 방식(toCenter == True시 라이더가 다시 center로 돌아가는 것 까지 운행 비용에 고려함)
    :return: (1)expected_cts: 선택되는 고객 이름 list ,
             (2)add_info [[선택하는 라이더 이름, 선택 된 고객 이름],...]
    """
    #print('rider_names', rider_names)
    #print('d_orders_res',d_orders_res, sorted(d_orders_res))
    #input('check')
    expected_cts = []
    customers = Basic.UnloadedCustomer(customer_set, now_time)
    already_selected = []  # 이미 선택되었을 고객.
    add_info = []
    for index in sorted(d_orders_res):
        test = []
        rider_name = rider_names[d_orders_res.index(index)]
        #print('rider_name',rider_name)
        rider = rider_set[rider_name]
        last_location = rider.now_ct[1]
        print('T {} 플랫폼 선택 시점의 설정 {} {} {}'.format( rider.env.now , toCenter, who, rider_route_cal_type))
        ct_infos = Basic.PriorityOrdering(rider, customers, toCenter=toCenter, who=who, rider_route_cal_type= rider_route_cal_type, last_location = last_location)
        print('플랫폼 예상 위치 {} :: 정보 확인{}'.format(rider.now_ct[1],ct_infos))
        if len(ct_infos) > 0:
            info = ct_infos[0]
            if info[1] > 0:
                if info[0] not in already_selected:
                    expected_cts.append(info[0])
                    already_selected.append(info[0])
                    add_info.append([rider_name, info[0], info[1]])
                    test.append([rider_name, ct_infos])
                    customers.remove(customer_set[info[0]])
            else:
                expected_cts.append(None)
                already_selected.append(None)
                add_info.append([rider_name, None, None])
        """
        for info in ct_infos:
            if info[0] not in already_selected and info[1] > 0:
                expected_cts.append(info[0])
                already_selected.append(info[0])
                add_info.append([rider_name, info[0], info[1]])
                test.append([rider_name, ct_infos])
                customers.remove(customer_set[info[0]])
                break        
        """
        if print_para == True:
            print('라이더 예상 고객은?',rider_name ,"::",test)
    return expected_cts, add_info

def SystemRunner(env, rider_set, customer_set, run_time, interval=10, No_subsidy=False, subsidy_offer=[],
                 subsidy_offer_count=[], time_thres=0.8, upper=10000, checker=False, toCenter=True, dummy_customer_para = False):
    """
    입력 값에 따라 시뮬레이션 진행
    No_subsidy에 따라 2가지 상황이 가능.
    if No_subsidy == True:
    ->보조금 지급을 하지 않는 상황
    else: No_subsidy == False
    ->보조금을 지급하는 상황
    :param env: simpy Environment
    :param rider_set: 라이더 dict <- 생성된 라이더가 저장되는 dict
    :param customer_set: 고객 dict <- 생성된 고객이 저장되는 dict
    :param cool_time: 시뮬레이션 실험 시간(분)
    :param interval: SubsidyAllocation 문제 실행 간격
    :param No_subsidy: 실험 파라메터
    :param subsidy_offer: 제안된 보조금 내역 저장소
    :param subsidy_offer_count: 제안된 보조금 횟수 계산
    :param time_thres: 임박한 고객을 계산하는데 사용되는 파라메터 <-WhoGetPriority
    :param upper: 지급하려는 보조금의 상한
    :param checker: SubsidyAllocation 문제 실행 간격 마다 결과를 확인하는 파라메터. (if True : 연산 멈춤 /if False: 프린트 문 출력)
    :param toCenter: PriorityOrdering에서 라이더의 가치를 판단하는 방식(toCenter == True시 라이더가 다시 center로 돌아가는 것 까지 운행 비용에 고려함)
    :param dummy_customer_para: SubsidyAllocation 문제 정의 시, 라이더가 고객을 선택하지 않는 경우를 표현하는 파라메터. (if True : 더미 고객 정의 /if False: 더미 고객 정의 X)
    """
    while env.now <= run_time:
        # 보조금 문제를 풀 필요가 없는 경우에는 문제를 풀지 않아야 한다.
        # C_p에 해당하는 고객이 이미 선택될 것으로 예상되는 경우
        # 라이더 체크 확인
        un_cts = Basic.UnloadedCustomer(customer_set, env.now)  # 아직 실리지 않은 고객 식별
        v_old, rider_names, cts_name, d_orders_res, times, end_times = ProblemInput(rider_set, customer_set, env.now, upper=upper, dummy_customer_para = dummy_customer_para)  # 문제의 입력 값을 계산
        urgent_cts, tem1 = Basic.WhoGetPriority(un_cts, len(rider_names), env.now, time_thres=time_thres)  # C_p 계산
        expected_cts, dummy = ExpectedSCustomer(rider_set, rider_names, d_orders_res, customer_set, round(env.now,2) , toCenter = toCenter, who = 'platform')
        if sorted(urgent_cts) == sorted(expected_cts) or len(urgent_cts) == 0 or len(rider_names) <= 1 or len(
                cts_name) <= 1 or No_subsidy == True:
            print('IP 풀이X', env.now,'급한 고객 수:', len(urgent_cts), '// 예상 매칭 고객 수:', expected_cts)
            print('가능한 라이더수:', len(rider_names), '//고객 수:', len(cts_name), '//No_subsidy:', No_subsidy)
            if No_subsidy == False:
                print('가상 매칭 결과', sorted(urgent_cts), sorted(expected_cts), 'No_subsidy', No_subsidy)
            # 문제를 풀지 않아도 서비스가 필요한 고객들이 모두 서비스 받을 수 있음.
            pass
        else:  # peak para 는 항상 참인 파라메터
            print('IP 풀이O', env.now,'급한 고객 수:', len(urgent_cts),'급한고객',urgent_cts,'// 예상 매칭 고객 수:', expected_cts, '//라이더 순서', d_orders_res)
            print('가능한 라이더수:', len(rider_names), '//고객 수:', len(cts_name), '//No_subsidy:', No_subsidy)
            print('V_old', np.shape(v_old), '//Time:', np.shape(times), '//EndTime:', np.shape(end_times))
            res, vars = lpg.LinearizedSubsidyProblem(rider_names, cts_name, v_old, d_orders_res, times, end_times,
                                                     lower_b=0, sp=urgent_cts, print_gurobi=False, upper_b=upper)
            if res == False:
                time_con_num = list(range(0, len(urgent_cts)))
                time_con_num.sort(reverse=True)
                try_num = 0
                for num in time_con_num:
                    res, vars = lpg.LinearizedSubsidyProblem(rider_names, cts_name, v_old, d_orders_res, times,
                                                             end_times, lower_b=0, sp=urgent_cts, print_gurobi=False,
                                                             relax=num, upper_b=upper)
                    try_num += 1
                    if res != False:
                        print('Relaxing Try #', time_con_num.index(num))
                        break
                print("Try#", try_num, '//So done', len(urgent_cts) - try_num)
            if res != False:
                feasibility, res2 = solver_interpreterForIP2(res[1])
                if feasibility == True:
                    print('Fee updater')
                    FeeUpdater(res2, customer_set, rider_names, rider_set, cts_name, env.now,
                               subsidy_offer=subsidy_offer, subsidy_offer_count=subsidy_offer_count, upper=upper)

        yield env.timeout(interval)
        # 보조금 초기화
        Basic.InitializeSubsidy(customer_set) # 보조금 초기화
        Basic.DefreezeAgent(rider_set, type = 'rider') #라이더 반영
        Basic.DefreezeAgent(customer_set, type = 'customer') #고객 반영
        if checker == False:
            print('Sys Run/T:' + str(env.now))
        else:
            input('Sys Run/T:' + str(env.now))