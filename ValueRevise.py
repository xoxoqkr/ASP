# -*- coding: utf-8 -*-
import numpy as np
import operator
import copy
import LinearizedASP_gurobi as lpg
import Basic_class as Basic
import SubsidyPolicy_class



def InputCalculate(expectation, actual_result):
    R = []
    E = []
    #플랫폼이 예상한 선택 정보
    for info in expectation:
        if info == 1:
            E.append(1)
        else:
            E.append(0)
    #라이더가 실제 선택한 정보
    for info in actual_result:
        if info == 1:
            R.append(1)
        else:
            R.append(0)
    return R, E
#[int(env.now), ct_name, self.last_location , rev_infos]
# rev_infos = [고객 이름, fee-cost - cost2, cost, fee]

def InputCalculate1(rider, customer_set, index = -1, LP_type = 'LP1'):
    """
    select_coefficient = []
    select_info = rider.choice_info[3][0]
    dist_cost = select_info[2]
    customer_type = customer_set[select_info[0]].type
    type_cost = rider.expect[customer_type] * 100
    fee = select_info[3]
    select_coefficient = [dist_cost, type_cost, fee]
    #다른 고객들에 대하여
    other_coefficient = []
    """
    rev_infos = []
    check = []
    print('InputCalculate2_index_정보;{}'.format(index))
    #try:
    #    print('info 정상 {}'.format(rider.choice_info[index][3]))
    #except:
    #    input('info 에러 {}'.format(rider.choice_info[index]))
    if len(rider.choice_info) == 0 or len(rider.choice_info[index]) < 3:
        input('정보 확인 3333; {}'.format(rider.choice_info[index]))
        return [], []
    for info in rider.choice_info[index][3]:
        #print('정보 확인 1234 ;; {}'.format(rider.choice_info[index]))
        ct_name = info[0]
        dist_cost = info[2]
        #try:
        #    print('에러 없음 {} ; {}; {}'.format(len(customer_set), ct_name, info))
        #    customer_type = customer_set[ct_name].type
        #except:
        #    print('에러 발생 {} ; {}; {}'.format(len(customer_set), ct_name, info))
        type_cost = rider.expect[info[11]]
        #type_cost = rider.expect[customer_type]
        fee = info[3]
        #fee = customer_set[ct_name].fee[0]
        if LP_type == 'LP1':
            rev_infos.append([-dist_cost, -type_cost, fee,ct_name,info[1],info[-1],-type_cost*rider.LP1p_coeff[1],-type_cost*rider.LP1p_coeff[1]])
        else:
            rev_infos.append([-dist_cost, -type_cost, fee,ct_name,info[1],info[-1],-type_cost*rider.LP2p_coeff[1],-type_cost*rider.LP2p_coeff[1]])
        check.append([ct_name, info[1]])
    check.sort(key = operator.itemgetter(1), reverse = True)
    print('Rider',rider.coeff)
    print('E_Rider LP1{} LP2 {} LP3{}'.format(rider.LP1p_coeff, rider.LP2p_coeff, rider.LP3p_coeff))
    print('Rev',rev_infos)
    if rev_infos[0][3] == check[0][0]:
        print('기존 p로 만족하는 데이터')
    else:
        print('기존 p로 만족X하는 데이터')
    #input('확인')
    return rev_infos[0], rev_infos[1:]


def InputCalculate2(rider, customer_set, index = -1, LP_type = 'rider_org_coeff',divider = [1.0,1.0]):
    """
    select_coefficient = []
    select_info = rider.choice_info[3][0]
    dist_cost = select_info[2]
    customer_type = customer_set[select_info[0]].type
    type_cost = rider.expect[customer_type] * 100
    fee = select_info[3]
    select_coefficient = [dist_cost, type_cost, fee]
    #다른 고객들에 대하여
    other_coefficient = []
    """
    rev_infos = []
    check = []
    print('InputCalculate2_index_정보;{}'.format(index))
    #try:
    #    print('info 정상 {}'.format(rider.choice_info[index][3]))
    #except:
    #    input('info 에러 {}'.format(rider.choice_info[index]))
    if len(rider.choice_info) == 0 or len(rider.choice_info[index]) < 3:
        input('정보 확인 3333; {}'.format(rider.choice_info[index]))
        return [], []
    for info in rider.choice_info[index][3]:
        #print('정보 확인 1234 ;; {}'.format(info))
        ct_name = info[0]
        dist_cost = info[6]/divider[0]
        #try:
        #    print('에러 없음 {} ; {}; {}'.format(len(customer_set), ct_name, info))
        #    customer_type = customer_set[ct_name].type
        #except:
        #    print('에러 발생 {} ; {}; {}'.format(len(customer_set), ct_name, info))
        type_cost = rider.expect[info[11]]
        #type_cost = rider.expect[customer_type]
        fee = info[3]/divider[1]
        #fee = customer_set[ct_name].fee[0]
        if LP_type == 'LP1':
            rev_infos.append([-dist_cost*rider.LP1p_coeff[0], -type_cost*rider.LP1p_coeff[1], fee*rider.LP1p_coeff[2],ct_name,info[1],info[-1],-type_cost*rider.LP1p_coeff[1],-type_cost*rider.LP1p_coeff[1]])
        if LP_type == 'LP2':
            rev_infos.append([-dist_cost*rider.LP2p_coeff[0], -type_cost*rider.LP2p_coeff[1], fee*rider.LP2p_coeff[2],ct_name,info[1],info[-1],-type_cost*rider.LP2p_coeff[1],-type_cost*rider.LP2p_coeff[1]])
        if LP_type == 'LP3':
            rev_infos.append([-dist_cost*rider.LP3p_coeff[0], -type_cost*rider.LP3p_coeff[1], fee*rider.LP3p_coeff[2],ct_name,info[1],info[-1],-type_cost*rider.LP3p_coeff[1],-type_cost*rider.LP3p_coeff[1]])
        else:
            rev_infos.append([-dist_cost, -type_cost, fee,ct_name,info[1],info[-1],-type_cost*rider.LP2p_coeff[1],-type_cost*rider.LP2p_coeff[1]])
        #print('확인',rev_infos)
        #rev_infos.append([-dist_cost, -type_cost, fee, ct_name, info[1], info[-1], -type_cost * rider.LP2p_coeff[1],-type_cost * rider.LP2p_coeff[1]])
        check.append([ct_name, info[1]])
    check.sort(key = operator.itemgetter(1), reverse = True)
    print('Rider',rider.coeff)
    print('E_Rider', rider.p_coeff)
    print('Rev',rev_infos)
    if rev_infos[0][3] == check[0][0]:
        print('기존 p로 만족하는 데이터')
    else:
        print('기존 p로 만족X하는 데이터')
    #input('확인')
    return rev_infos[0], rev_infos[1:]

def ChoiceCheck(rider, customer_set):
    """
    라이더가 고객 선택을 제대로 한 것인지 확인.
    :param rider:
    :return:
    """
    #1 자신이 선택할 수 있었던 것 중 최고를 선택하는가?
    print('rider',rider.name,'.choice_info',rider.choice_info)
    for raw_info in rider.choice_info:
        info = raw_info[3]
        print('info',info)
        select = info[0]
        others = info[1:]
        #select_score =
        #print('select',select)
        #print('others',others)
        other_score = []
        for other in [select]+others:
            coeff_index = 2
            value = 0
            value += -other[2]* rider.p_coeff[0]
            value += -customer_set[other[0]].type * rider.p_coeff[1]
            value += other[3] * rider.p_coeff[2]
            """
            for coeff in rider.p_coeff[:2]:
                value += other[coeff_index]*coeff
                coeff_index += 1
            """
            other_score.append([other[0],round(value,1)]) #[[고객 이름, 이윤],...]
        #print('P예상 선택함',other_score[0])
        #other_score = other_score[1:]
        other_score.sort(key=operator.itemgetter(1))
        other_score.reverse()
        print('P예상 선택함',other_score[0],'P예상 후보들', other_score)
        #실제 계산 부.
        actual_score = []
        for other in [select]+others:
            coeff_index = 2
            value = 0
            value += -other[2]* rider.coeff[0]
            value += -customer_set[other[0]].type * rider.coeff[1]
            value += other[3] * rider.coeff[2]
            """
            for coeff in rider.p_coeff[:2]:
                value += other[coeff_index]*coeff
                coeff_index += 1
            """
            actual_score.append([other[0],round(value,1)])
        #print('R예상 선택함',actual_score)
        #actual_score = actual_score[1:]
        actual_score.sort(key=operator.itemgetter(1))
        actual_score.reverse()
        print('R예상 후보들', actual_score)


    return None


def Comapre2List(base, compare, ox_table):
    #base에는 있으나, compare는 없는 요소 찾기.
    base.sort(key=operator.itemgetter(0))
    compare.sort(key=operator.itemgetter(0))
    expectation = []
    actual = []
    for info1 in base:
        ox_table[0] += 1
        for info2 in compare:
            if info1[0] == info2[0]:
                if info1[1] == info2[1]:
                    ox_table[1] += 1
                    ox_table[3] += 1
                    pass
                else:
                    expectation.append(info1)
                    actual.append(info2)
                break
    ox_table[2] += len(compare)
    return expectation, actual


def RiderChoiceWithInterval(rider_set, rider_names, now, interval = 10):
    """
    [now - interval, now)까지의 라이더가 선택한 고객을 반환
    [[라이더이름, 고객이름],...]
    :param rider_set:
    :param rider_names:
    :param now:
    :param interval:
    :return:
    """
    #print(now - interval,'->',now, "라이더", rider_names)
    actual_choice = []
    for rider_name in rider_names:
        rider = rider_set[rider_name]
        #print('rider.choice', rider.choice)
        for info in rider.choice:
            #print(info[1])
            if now - interval < info[1] <= now:
                actual_choice.append([rider.name, info[0]])
                print("라이더의 선택",[rider.name, info[0]])
                break
            else:
                pass
    return actual_choice


def SolverBlock(rider, selected, others, customer_set, weight_sum = True, LP_type = 'LP1', beta = 1):
    indexs = list(range(len(rider.choice_info) - 1))
    indexs.reverse()
    end_count = len(indexs)
    if LP_type == 'LP1':
        p_coeff = rider.LP1p_coeff
        selected_value = np.dot(rider.p_coeff, selected[:len(rider.p_coeff)])
    elif LP_type == 'LP2':
        p_coeff = rider.LP2p_coeff
        selected_value = np.dot(rider.p_coeff, selected[:len(rider.p_coeff)])
        end_count = max(1,int(len(indexs) * beta - len(rider.violated_choice_info)))
    elif LP_type == 'LP3':
        p_coeff = rider.LP3p_coeff
        selected_value = np.dot(rider.p_coeff, selected[:len(rider.p_coeff)])
    elif LP_type == 'LP3_2':
        p_coeff = rider.LP3_2p_coeff
        selected_value = np.dot(rider.p_coeff, selected[:len(rider.p_coeff)])
    else:
        input('SolverBlock 에러 발생')
        selected_value = 0
    past_choices = []
    satisfy = True
    #input('체크0')
    for other_info in others:
        if selected_value < np.dot(p_coeff, other_info[:len(rider.p_coeff)]):
            satisfy = False
            break
    #input('체크0-1')
    if satisfy == True:
        feasibility = False
        return feasibility, None, None, None, None
    else:
        #input('체크0-3')
        error_para = False
        added_past_data = []
        #input('체크1-1')
        if LP_type == 'LP2':
            added_past_data += rider.violated_choice_info[:len(rider.violated_choice_info) - 1]
        added_past_data += indexs[:end_count]
        #input('체크1-2')
        for index1 in added_past_data:
            if index1 not in range(len(rider.choice_info)):
                # input('탐색 에러 발생 {} ;; {}'.format(index1, len(rider.choice_info)))
                past_select = []
                past_others = []
                error_para = True
            else:
                past_select, past_others = InputCalculate2(rider, customer_set,
                                                           index=index1,
                                                           LP_type='LP2')  # 실제 라이더가 선택할 시점의 [-dist_cost, -type_cost, fee]
            if len(past_others) > 0:
                past_choices.append([past_select, past_others])
        #input('체크2')
        if LP_type == 'LP1':
            feasibility, res, exe_t, obj = lpg.ReviseCoeffAP1(selected, others, rider.LP1p_coeff,past_data=past_choices,weight_sum=weight_sum)
        elif LP_type == 'LP2':
            feasibility, res, exe_t, obj = lpg.ReviseCoeffAP2(selected, others, rider.LP2p_coeff,past_data=past_choices,weight_sum=weight_sum)
        elif LP_type == 'LP3':
            feasibility, res, exe_t, obj = lpg.ReviseCoeffAP3(selected, others, rider.LP3p_coeff,past_data=past_choices,weight_sum=weight_sum)
        elif LP_type == 'LP3_2':
            feasibility, res, exe_t, obj = lpg.ReviseCoeffAP3(selected, others, rider.LP3_2p_coeff,past_data=past_choices,weight_sum=weight_sum, solution_space_cut= True)
        else:
            pass
        input('{};{};{};{};{};{};'.format(LP_type, feasibility, res, exe_t, obj, len(past_choices)))
        if feasibility == True:
            return feasibility, res, exe_t, obj, len(past_choices)
        else:
            return feasibility, None, None, None, None


def SolutionSaveBlock(feasibility, res, exe_t, obj, rider, past_data_num, LP_type = 'LP1'):
    if feasibility == True:
        revise_value = 0
        for index in range(len(res)):
            if LP_type == 'LP1':
                rider.LP1p_coeff[index] = res[index]
                revise_value = 1
            elif LP_type == 'LP2':
                revise_value += abs(res[index])
                rider.LP2p_coeff[index] += res[index]
            elif LP_type == 'LP3':
                rider.LP3p_coeff[index] = res[index]
                revise_value = 1
            elif LP_type == 'LP3_2':
                pass
        if revise_value > 0:
            input('결과 기록')
            if LP_type == 'LP1':
                rider.LP1p_coeff[index] = res[index]
                rider.LP1History.append(copy.deepcopy(rider.LP1p_coeff) + [copy.deepcopy(len(rider.choice)), past_data_num] + [exe_t, obj])
            elif LP_type == 'LP2':
                rider.LP2p_coeff[index] += res[index]
                rider.LP2History.append(copy.deepcopy(rider.LP2p_coeff) + [copy.deepcopy(len(rider.choice)), past_data_num] + [exe_t, obj])
            elif LP_type == 'LP3':
                rider.LP3p_coeff[index] = res[index]
                rider.LP3History.append(copy.deepcopy(rider.LP3p_coeff) + [copy.deepcopy(len(rider.choice)), past_data_num] + [exe_t, obj])
            elif LP_type == 'LP3_2':
                rider.LP3_2p_coeff[index] = res[index]
                rider.LP3_2History.append(copy.deepcopy(rider.LP3_2p_coeff) + [copy.deepcopy(len(rider.choice)), past_data_num] + [exe_t, obj])
    else:
        pass


def IncentiveForValueWeightExpectationLP3(rider, customer_set, LP_type = 'LP3', upper = 1000, slack1 = 1):
    #현재는 LP2로 진행
    un_assigned_cts = []
    for customer_name in customer_set:
        customer = customer_set[customer_name]
        if customer.time_info[1] == None:
            un_assigned_cts.append(customer)
    infos = Basic.PriorityOrdering(rider, un_assigned_cts, toCenter=False, who='test_rider',rider_route_cal_type='return', last_location=None, LP_type=LP_type)
    print('고객 정보 확인',infos)
    print_res = []
    if len(infos) > 1:
        selected_value = infos[0][1]
        for info in infos[1:]:
            diff = selected_value - info[1]
            print('diff', diff, selected_value, info[1])
            try:
                if rider.LP_type == 'LP1':
                    divider = rider.LP1p_coeff[2]
                elif rider.LP_type == 'LP2':
                    divider = rider.LP2p_coeff[2]
                elif rider.LP_type == 'LP3':
                    divider = rider.LP3p_coeff[2]
                else:
                    divider = 1
                    input('ERROR')
                required_incentive = (diff / divider)*slack1
                if 0 < required_incentive < upper : # and rider.LP3p_coeff[2] > 0:
                    customer_set[info[0]].fee[1] = required_incentive
                    customer_set[info[0]].fee[2] = 'all'
                    customer.fee_history.append(0)
                    print_res.append([info[0], required_incentive, customer_set[info[0]].time_info])
            except:
                pass
    #input('보조금 확인 {}'.format(print_res))


def SystemRunner(env, rider_set, customer_set, cool_time, ox_table ,interval=10, checker = False, toCenter = True, rider_route_cal_type = 'return',
                 weight_sum = False, revise = False, beta = 0, LP_type = 'LP1', validation_t = 100, incentive_time = 200, slack1 = 1):
    # 보조금 부문을 제외하고 단순하게 작동하는 것으로 시험해 볼 것.
    while env.now <= cool_time:
        #플랫폼이 예상한 라이더-고객 선택
        un_served_ct = Basic.UnloadedCustomer(customer_set, env.now)
        ava_rider = Basic.AvaRider(rider_set, env.now)
        print("시간::",env.now," / 서비스가 필요한 고객 수::", len(un_served_ct), " / interval동안 빈 차량 수::", len(ava_rider))
        v_old, rider_names, cts_name, d_orders_res, times, end_times = SubsidyPolicy_class.ProblemInput(rider_set, customer_set, env.now)  # 문제의 입력 값을 계산
        print("선택순서", d_orders_res)
        expected_cts, platform_expected = SubsidyPolicy_class.ExpectedSCustomer(rider_set, rider_names, d_orders_res, customer_set, round(env.now, 2), toCenter=toCenter,
                                                                                who='test_platform', rider_route_cal_type = rider_route_cal_type) # platform_expected = [[라이더 이름, 고객 이름],...,]
        #platform_expected = [[rider_name, 고객 이름],...,]
        for rider_name in rider_set:
            rider = rider_set[rider_name]
            if env.now < rider.end_time <= env.now + interval and env.now < incentive_time:
                print('라이더 :{} ; 고객 선택 시점 :{}; 현재 시점 :{}'.format(rider.name, rider.end_time, int(env.now)))
                #IncentiveForValueWeightExpectationLP3(rider, customer_set, LP_type='LP3', slack1 = slack1)
                pass
        yield env.timeout(interval)
        now = round(env.now, 1)
        print('과거 {} ~ 현재 시점 {}// 라이더 선택들{}'.format(now - interval, now, rider_set[0].choice))
        print('플랫폼이 예상한 라이더 선택 고객', expected_cts, platform_expected)
        #선택된 라이더들에 대한 가치 함수 갱신
        actual_choice = RiderChoiceWithInterval(rider_set, rider_names, now, interval=interval)
        expectation, actual = Comapre2List(platform_expected, actual_choice, ox_table)
        #ox_table_save
        print("차이발생/T now:",now,"/예상한 선택", expectation , "/실제 선택", actual, "/실제선택2:",actual_choice)

        if env.now > validation_t:
            if len(rider.choice_info) > 0 and env.now - interval <= rider.choice_info[-1][0]:
                print('R{} T:{}~{}에 고객 {} 선택'.format(rider_name, int(env.now - interval), int(env.now),
                                                     rider.choice_info[-1]))
                #input('주문 선택')
                selected, others = InputCalculate2(rider, customer_set)  # 실제 라이더가 선택한 고객의 [-dist_cost, -type_cost, fee]
                true_selected_value = np.dot(rider.coeff, selected[:len(rider.coeff)])
                rider.validations[3] += 1
                #LP1 블록
                LP1_res = True
                LP1_selected_value = np.dot(rider.LP1p_coeff, selected[:len(rider.LP1p_coeff)])
                for other_info in others:
                    if LP1_selected_value < np.dot(rider.LP1p_coeff, other_info[:len(rider.LP1p_coeff)]):
                        LP1_res = False
                        break
                if LP1_res == True:
                    rider.validations[0] += 1
                diff1 = true_selected_value - LP1_selected_value
                rider.validations_detail[0].append(diff1)
                rider.validations_detail_abs[0].append(abs(diff1))
                #LP2블록
                LP2_res = True
                LP2_selected_value = np.dot(rider.LP2p_coeff, selected[:len(rider.LP2p_coeff)])
                for other_info in others:
                    if LP2_selected_value < np.dot(rider.LP2p_coeff, other_info[:len(rider.LP2p_coeff)]):
                        LP2_res = False
                        break
                if LP2_res == True:
                    rider.validations[1] += 1
                diff2 = true_selected_value - LP2_selected_value
                rider.validations_detail[1].append(diff2)
                rider.validations_detail_abs[1].append(abs(diff2))
                #LP3블록
                LP3_res = True
                LP3_selected_value = np.dot(rider.LP3p_coeff, selected[:len(rider.LP3p_coeff)])
                for other_info in others:
                    if LP3_selected_value < np.dot(rider.LP3p_coeff, other_info[:len(rider.LP3p_coeff)]):
                        LP3_res = False
                        break
                if LP3_res == True:
                    rider.validations[2] += 1
                diff3 = true_selected_value - LP3_selected_value
                rider.validations_detail[2].append(diff3)
                rider.validations_detail_abs[2].append(abs(diff3))
        else:
            print('시작 확인')
            for rider_name in rider_set:
                rider = rider_set[rider_name]
                print('라이더:',rider.name)
                #print(rider.choice_info)
                if len(rider.choice_info) > 0 and env.now - interval <= rider.choice_info[-1][0]:
                    print('R{} T:{}~{}에 고객 {} 선택'.format(rider_name, int(env.now - interval), int(env.now), rider.choice_info[-1]))
                    selected, others = InputCalculate2(rider, customer_set)  # 실제 라이더가 선택한 고객의 [-dist_cost, -type_cost, fee]
                    indexs = list(range(len(rider.choice_info) - 1))
                    indexs.reverse()
                    #rider.validations[3] += 1
                    #if #LP_type == 'LP1':
                    #LP1블럭
                    """
                    LP1past_choices = []
                    for index1 in indexs:
                        past_select, past_others = InputCalculate2(rider, customer_set, index=index1, LP_type = 'LP1') #실제 라이더가 선택할 시점의 [-dist_cost, -type_cost, fee]
                        if len(past_others) > 0:
                            LP1past_choices.append([past_select, past_others])
                    LP1feasibility, LP1res, LP1exe_t = lpg.ReviseCoeffAP1(selected, others, [0,0,0], past_data=LP1past_choices,
                                                          weight_sum=weight_sum)                    
                    """
                    #elif True:

                    # LP1도 기존의 해가 만족하지 못할 때 수행
                    LP1past_choices = []
                    LP1_satisfy = True
                    LP1_selected_value = np.dot(rider.LP1p_coeff,selected[:len(rider.LP1p_coeff)])
                    for other_info in others:
                        if LP1_selected_value < np.dot(rider.LP1p_coeff,other_info[:len(rider.LP1p_coeff)]):
                            LP1_satisfy = False
                            print('LP1 틀림', LP1_selected_value, '<',
                                  np.dot(rider.LP1p_coeff, other_info[:len(rider.LP1p_coeff)]))
                            break
                        else:
                            #rider.validations[0] += 1
                            pass
                    #LP1_satisfy = False
                    if LP1_satisfy == True:
                        LP1feasibility = False
                    else:
                        for index1 in indexs:
                            past_select, past_others = InputCalculate2(rider, customer_set, index=index1,
                                                                       LP_type='LP1')  # 실제 라이더가 선택할 시점의 [-dist_cost, -type_cost, fee]
                            if len(past_others) > 0:
                                LP1past_choices.append([past_select, past_others])
                        LP1feasibility, LP1res, LP1exe_t, LP1_obj = lpg.ReviseCoeffAP1(selected, others, rider.LP1p_coeff,
                                                                              past_data=LP1past_choices,
                                                                              weight_sum=weight_sum)
                    #LP2블럭
                    LP2past_choices = []
                    LP2_satisfy = True
                    LP2_selected_value = np.dot(rider.LP2p_coeff,selected[:len(rider.LP2p_coeff)])
                    for other_info in others:
                        if LP2_selected_value < np.dot(rider.LP2p_coeff,other_info[:len(rider.LP2p_coeff)]):
                            rider.violated_choice_info.append(copy.deepcopy(len(rider.choice_info)))
                            LP2_satisfy = False
                            print('LP2 틀림',LP2_selected_value,'<',np.dot(rider.LP2p_coeff,other_info[:len(rider.LP2p_coeff)]))
                            break
                        else:
                            #rider.validations[1] += 1
                            pass
                    if LP2_satisfy == True:
                        LP2feasibility = False
                    else:
                        error_para = False
                        for index1 in rider.violated_choice_info[:len(rider.violated_choice_info)-1] + indexs[:max(1, int(len(indexs) * beta - len(rider.violated_choice_info)))]:
                            if index1 not in range(len(rider.choice_info)):
                                #input('탐색 에러 발생 {} ;; {}'.format(index1, len(rider.choice_info)))
                                past_select = []
                                past_others = []
                                error_para = True
                            else:
                                past_select, past_others = InputCalculate2(rider, customer_set,
                                                                           index=index1, LP_type= 'LP2')  # 실제 라이더가 선택할 시점의 [-dist_cost, -type_cost, fee]
                            if len(past_others) > 0:
                                LP2past_choices.append([past_select, past_others])
                        if error_para == True:
                            #input('데이터 확인 {}'.format(LP2past_choices))
                            pass
                        LP2feasibility, LP2res, LP2exe_t, LP2_obj = lpg.ReviseCoeffAP2(selected, others, rider.LP2p_coeff, past_data=LP2past_choices,
                                                              weight_sum=weight_sum)
                    #LP3블럭
                    LP3past_choices = []
                    LP3_satisfy = True
                    LP3_selected_value = np.dot(rider.LP3p_coeff,selected[:len(rider.LP3p_coeff)])
                    check_list = [LP3_selected_value]
                    for other_info in others:
                        check_list.append(np.dot(rider.LP3p_coeff, other_info[:len(rider.LP3p_coeff)]))
                        if LP3_selected_value < np.dot(rider.LP3p_coeff,other_info[:len(rider.LP3p_coeff)]):
                            LP3_satisfy = False
                            break
                        else:
                            #rider.validations[2] += 1
                            pass
                    #input('선택 보조금 확인 {}'.format(check_list))
                    LP3_satisfy = False #todo : LP는 계속 풀리도록
                    if LP3_satisfy == True:
                        LP3feasibility = False
                    else:
                        for index1 in indexs:
                            past_select, past_others = InputCalculate2(rider, customer_set, index=index1,
                                                                       LP_type='LP3')  # 실제 라이더가 선택할 시점의 [-dist_cost, -type_cost, fee]
                            if len(past_others) > 0:
                                LP3past_choices.append([past_select, past_others])
                        LP3feasibility, LP3res, LP3exe_t, LP3_obj = lpg.ReviseCoeffAP3_WIP(selected, others, rider.LP3p_coeff,
                                                                              past_data=LP3past_choices,
                                                                              weight_sum=weight_sum)
                        #LP3feasibility, LP3res, LP3exe_t, LP3_obj = lpg.ReviseCoeffAP3(selected, others, rider.LP3p_coeff,
                        #                                                      past_data=LP3past_choices,
                        #                                                      weight_sum=weight_sum)
                    input('LP1{},LP2{},LP3{}'.format(LP1_satisfy,LP2_satisfy,LP3_satisfy))
                    if LP1feasibility == True:
                        LP1revise_value = 0
                        for index in range(len(LP1res)):
                            #rider.p_coeff[index] = res[index]
                            rider.LP1p_coeff[index] = LP1res[index]
                            LP1revise_value = 1
                        if LP1revise_value > 0:
                            rider.LP1History.append(
                                copy.deepcopy(rider.LP1p_coeff) + [copy.deepcopy(len(rider.choice)), len(LP1past_choices)] + [
                                    LP1exe_t, LP1_obj])
                    if LP2feasibility == True:
                        LP2revise_value = 0
                        for index in range(len(LP2res)):
                            rider.LP2p_coeff[index] += LP2res[index]
                            LP2revise_value += abs(LP2res[index])
                        if LP2revise_value > 0:
                            rider.LP2History.append(copy.deepcopy(rider.LP2p_coeff)+[copy.deepcopy(len(rider.choice)),len(LP2past_choices)] +[LP2exe_t, LP2_obj])
                    if LP3feasibility == True:
                        LP3revise_value = 0
                        for index in range(len(LP3res)):
                            rider.LP3p_coeff[index] = LP3res[index]
                            LP3revise_value = 1
                        if LP3revise_value > 0:
                            #input('LP3 기록')
                            rider.LP3History.append(copy.deepcopy(rider.LP3p_coeff) + [copy.deepcopy(len(rider.choice)), len(LP3past_choices)] + [LP3exe_t, LP3_obj])
                    """
                    if LP3_2feasibility == True:
                        LP3_2revise_value = 0
                        for index in range(len(LP3_2res)):
                            rider.LP3_2p_coeff[index] = LP3_2res[index]
                            LP3_2revise_value = 1
                        if LP3_2revise_value > 0:
                            #input('LP3 기록')
                            rider.LP3_2History.append(copy.deepcopy(rider.LP3_2p_coeff) + [copy.deepcopy(len(rider.choice)), len(LP3past_choices)] + [LP3_2exe_t, LP3_2_obj])
                    
                    #개선한 코드
                    feasibility, res, exe_t, obj,past_data_num = SolverBlock(rider, selected, others, customer_set, weight_sum=True, LP_type='LP1', beta=1)
                    SolutionSaveBlock(feasibility, res, exe_t, obj, rider, past_data_num, LP_type='LP1')
                    feasibility, res, exe_t, obj,past_data_num = SolverBlock(rider, selected, others, customer_set, weight_sum=True, LP_type='LP2', beta=1)
                    SolutionSaveBlock(feasibility, res, exe_t, obj, rider, past_data_num, LP_type='LP2')
                    feasibility, res, exe_t, obj,past_data_num = SolverBlock(rider, selected, others, customer_set, weight_sum=True, LP_type='LP3', beta=1)
                    SolutionSaveBlock(feasibility, res, exe_t, obj, rider, past_data_num, LP_type='LP3')                    
                    """
                    print('원래 결과', rider.coeff)
                    print(LP1feasibility,'LP1:', rider.LP1p_coeff, '에러 확인',sum(rider.LP1p_coeff))
                    print(LP2feasibility,'LP2:', rider.LP2p_coeff, '에러 확인',sum(rider.LP2p_coeff))
                    print(LP3feasibility,'LP3:', rider.LP3p_coeff, '에러 확인',sum(rider.LP3p_coeff))
                    if LP1feasibility == True or LP2feasibility == True or LP3feasibility == True:
                        #input('해 확인')
                        pass
            #print('예상과 동일한 선택 수행/ T:{}'.format(int(env.now)))

        # 보조금 초기화
        Basic.InitializeSubsidy(customer_set)  # 보조금 초기화
        Basic.DefreezeAgent(rider_set, type='rider')  # 라이더 반영
        Basic.DefreezeAgent(customer_set, type='customer')  # 고객 반영
        if checker == False:
            print('가치함수 갱신 Sys Run/T:' + str(env.now))
        else:
            input('가치함수 갱신 Sys Run/T:' + str(env.now))
        #input("정보확인1")


def LP_Solver(rider, customer_set, p_coefff, LP_type = 'init', trigger_type = 'init', weight_sum = True, selected_nonnegative = True):
    #(1) 트리거
    LP_feasibility = None
    trigger = None
    res = None
    obj = None
    exe_t = None
    selected, others = InputCalculate2(rider, customer_set, divider=[rider.coeff[0], rider.coeff[2]])  # 실제 라이더가 선택한 고객의 [-dist_cost, -type_cost, fee]
    print('선택 고객', selected)
    if trigger_type == 'Always':
        trigger = True
    elif trigger_type == 'Conditional':
        selected_value = np.dot(p_coefff, selected[:len(p_coefff)])
        for other_info in others:
            if selected_value < np.dot(p_coefff, other_info[:len(p_coefff)]):
                trigger = False
                break
            else:
                # rider.validations[0] += 1
                pass
    else:
        input('LP_Solver error1')
    if trigger == True:
        indexs = list(range(len(rider.choice_info) - 1))
        indexs.reverse()
        past_choices = []
        for index1 in indexs:
            past_select, past_others = InputCalculate2(rider, customer_set, index=index1,divider=[rider.coeff[0], rider.coeff[2]])  # 실제 라이더가 선택할 시점의 [-dist_cost, -type_cost, fee]
            if len(past_others) > 0:
                past_choices.append([past_select, past_others])
        if LP_type == 'LP1':
            LP_feasibility, res, exe_t, obj = lpg.ReviseCoeffAP1(selected, others, p_coefff,
                                                                           past_data=past_choices,
                                                                           weight_sum=weight_sum)
        elif LP_type == 'LP2':
            LP_feasibility, res, exe_t, obj = lpg.ReviseCoeffAP2(selected, others, p_coefff,
                                                                 past_data=past_choices,
                                                                 weight_sum=weight_sum)
        elif LP_type == 'LP3':
            LP_feasibility, res, exe_t, obj = lpg.ReviseCoeffAP3(selected, others, p_coefff,
                                                                 past_data=past_choices,
                                                                 weight_sum=weight_sum, selected_nonnegative = selected_nonnegative)
        else:
            input('LP_Solver error2')
    elif trigger == False:
        LP_feasibility = False
        pass
    else:
        input('LP_Solver error3')
    return LP_feasibility, res, exe_t, obj


def RiderWeightUpdaterByLPSolution(rider, res, exe_t, obj, LP_type = 'init', now_t = 0):
    #1 계산 된 자료 저장 부분
    len_past_choice = copy.deepcopy(len(rider.choice))
    common_info = [len_past_choice, len_past_choice - 1 ,exe_t, obj, now_t]
    if LP_type == 'LP1':
        rider.LP1History.append(copy.deepcopy(rider.LP1p_coeff) + common_info)
    elif LP_type == 'LP2':
        rider.LP2History.append(copy.deepcopy(rider.LP2p_coeff) + common_info)
    elif LP_type == 'LP3':
        rider.LP3History.append(copy.deepcopy(rider.LP3p_coeff) + common_info)
    else:
        input('RiderWeightUpdaterByLPSolution error1')
    #2 갱신 된 가중치 반영 하는 부분
    if LP_type == 'LP1':
        for index in range(len(res)):
            rider.LP1p_coeff[index] = res[index]
    elif LP_type == 'LP3':
        for index in range(len(res)):
            rider.LP3p_coeff[index] = res[index]
    elif LP_type == 'LP2':
        for index in range(len(res)):
            rider.LP2p_coeff[index] += res[index]
    else:
        input('RiderWeightUpdaterByLPSolution error1')


def RiderWeightUpdater(rider, customer_set,weight_sum, beta = 1, LP_type = 'LP3'):
    print('시작 확인 :: 라이더:', rider.name)
    #selected, others = InputCalculate2(rider,customer_set)  # 실제 라이더가 선택한 고객의 [-dist_cost, -type_cost, fee]
    selected, others = InputCalculate2(rider, customer_set, divider=[rider.coeff[0], rider.coeff[2]])
    indexs = list(range(len(rider.choice_info) - 1))
    indexs.reverse()
    # LP1도 기존의 해가 만족하지 못할 때 수행
    LP1past_choices = []
    LP1_satisfy = True
    LP1_selected_value = np.dot(rider.LP1p_coeff, selected[:len(rider.LP1p_coeff)])
    print('LP1_selected_value ',LP1_selected_value)
    for other_info in others:
        if LP1_selected_value < np.dot(rider.LP1p_coeff, other_info[:len(rider.LP1p_coeff)]):
            LP1_satisfy = False
            print('LP1 틀림', LP1_selected_value, '<',
                  np.dot(rider.LP1p_coeff, other_info[:len(rider.LP1p_coeff)]))
            break
        else:
            # rider.validations[0] += 1
            pass
    #LP1블럭
    if LP1_satisfy == True:
        LP1feasibility = False
    else:
        for index1 in indexs:
            past_select, past_others = InputCalculate2(rider, customer_set, index=index1,
                                                                       divider = [rider.coeff[0],rider.coeff[2]],LP_type='LP1')  # 실제 라이더가 선택할 시점의 [-dist_cost, -type_cost, fee]
            if len(past_others) > 0:
                LP1past_choices.append([past_select, past_others])
        LP1feasibility, LP1res, LP1exe_t, LP1_obj = lpg.ReviseCoeffAP1(selected, others,
                                                                       rider.LP1p_coeff,
                                                                       past_data=LP1past_choices,
                                                                       weight_sum=weight_sum)
    # LP2블럭
    LP2past_choices = []
    LP2_satisfy = True
    LP2_selected_value = np.dot(rider.LP2p_coeff, selected[:len(rider.LP2p_coeff)])
    print('LP2_selected_value ', LP2_selected_value)
    for other_info in others:
        if LP2_selected_value < np.dot(rider.LP2p_coeff, other_info[:len(rider.LP2p_coeff)]):
            rider.violated_choice_info.append(copy.deepcopy(len(rider.choice_info)))
            LP2_satisfy = False
            print('LP2 틀림', LP2_selected_value, '<',
                  np.dot(rider.LP2p_coeff, other_info[:len(rider.LP2p_coeff)]))
            break
        else:
            # rider.validations[1] += 1
            pass
    if LP2_satisfy == True:
        LP2feasibility = False
    else:
        error_para = False
        for index1 in rider.violated_choice_info[:len(rider.violated_choice_info) - 1] + indexs[:max(1,
                                                                                                     int(len(indexs) * beta - len(
                                                                                                             rider.violated_choice_info)))]:
            if index1 not in range(len(rider.choice_info)):
                # input('탐색 에러 발생 {} ;; {}'.format(index1, len(rider.choice_info)))
                past_select = []
                past_others = []
                error_para = True
            else:
                past_select, past_others = InputCalculate2(rider, customer_set,divider = [rider.coeff[0],rider.coeff[2]],
                                                                           index=index1, LP_type= 'LP2')  # 실제 라이더가 선택할 시점의 [-dist_cost, -type_cost, fee]
            if len(past_others) > 0:
                LP2past_choices.append([past_select, past_others])
        if error_para == True:
            # input('데이터 확인 {}'.format(LP2past_choices))
            pass
        LP2feasibility, LP2res, LP2exe_t, LP2_obj = lpg.ReviseCoeffAP2(selected, others,
                                                                       rider.LP2p_coeff,
                                                                       past_data=LP2past_choices,
                                                                       weight_sum=weight_sum)
    # LP3블럭
    LP3past_choices = []
    LP3_satisfy = True
    LP3_selected_value = np.dot(rider.LP3p_coeff, selected[:len(rider.LP3p_coeff)])
    check_list = [LP3_selected_value]
    for other_info in others:
        check_list.append(np.dot(rider.LP3p_coeff, other_info[:len(rider.LP3p_coeff)]))
        if LP3_selected_value < np.dot(rider.LP3p_coeff, other_info[:len(rider.LP3p_coeff)]):
            LP3_satisfy = False
            break
        else:
            # rider.validations[2] += 1
            pass
    # input('선택 보조금 확인 {}'.format(check_list))
    LP3_satisfy = False  # todo : LP는 계속 풀리도록
    if LP3_satisfy == True:
        LP3feasibility = False
    else:
        for index1 in indexs:
            past_select, past_others = InputCalculate2(rider, customer_set, index=index1,divider = [rider.coeff[0],rider.coeff[2]],
                                                                       LP_type='LP3')  # 실제 라이더가 선택할 시점의 [-dist_cost, -type_cost, fee]
            if len(past_others) > 0:
                LP3past_choices.append([past_select, past_others])
        LP3feasibility, LP3res, LP3exe_t, LP3_obj = lpg.ReviseCoeffAP3_WIP(selected, others,
                                                                           rider.LP3p_coeff,
                                                                           past_data=LP3past_choices,
                                                                           weight_sum=weight_sum)
        # LP3feasibility, LP3res, LP3exe_t, LP3_obj = lpg.ReviseCoeffAP3(selected, others, rider.LP3p_coeff,
        #                                                      past_data=LP3past_choices,
        #                                                      weight_sum=weight_sum)
    #input('LP1{},LP2{},LP3{}'.format(LP1_satisfy, LP2_satisfy, LP3_satisfy))
    if LP1feasibility == True:
        LP1revise_value = 0
        for index in range(len(LP1res)):
            # rider.p_coeff[index] = res[index]
            rider.LP1p_coeff[index] = LP1res[index]
            LP1revise_value = 1
        if LP1revise_value > 0:
            rider.LP1History.append(
                copy.deepcopy(rider.LP1p_coeff) + [copy.deepcopy(len(rider.choice)),
                                                   len(LP1past_choices)] + [
                    LP1exe_t, LP1_obj])
    if LP2feasibility == True:
        LP2revise_value = 0
        for index in range(len(LP2res)):
            rider.LP2p_coeff[index] += LP2res[index]
            LP2revise_value += abs(LP2res[index])
        if LP2revise_value > 0:
            rider.LP2History.append(copy.deepcopy(rider.LP2p_coeff) + [copy.deepcopy(len(rider.choice)),
                                                                       len(LP2past_choices)] + [
                                        LP2exe_t, LP2_obj])
    if LP3feasibility == True:
        LP3revise_value = 0
        for index in range(len(LP3res)):
            rider.LP3p_coeff[index] = LP3res[index]
            LP3revise_value = 1
        if LP3revise_value > 0:
            # input('LP3 기록')
            rider.LP3History.append(copy.deepcopy(rider.LP3p_coeff) + [copy.deepcopy(len(rider.choice)),
                                                                       len(LP3past_choices)] + [
                                        LP3exe_t, LP3_obj])

def RiderWeightUpdater2(rider, customer_set,weight_sum, LP_type = 'LP3', trigger_type = 'Always', selected_nonnegative = True):
    if LP_type == 'LP1':
        p_coefff = rider.LP1p_coeff
    elif LP_type == 'LP2':
        p_coefff = rider.LP2p_coeff
    elif LP_type == 'LP3':
        p_coefff = rider.LP3p_coeff
    else:
        pass
    LP_feasibility, res, exe_t, obj = LP_Solver(rider, customer_set, p_coefff, LP_type=LP_type,
                                                trigger_type=trigger_type, weight_sum=weight_sum,
                                                selected_nonnegative=selected_nonnegative)
    if LP_feasibility == True:
        RiderWeightUpdaterByLPSolution(rider, res, exe_t, obj, LP_type=LP_type, now_t=int(rider.env.now))
    elif LP_feasibility == False:
        #input('{} 불가능해 발생'.format(LP_type))
        pass
    else:
        input('LP_Solver 잘못된 입렵 값')

"""
def SystemRunner2(env, rider_set, customer_set, cool_time, init_coeff = [0.5,0.5], interval=10, checker = False):
    #박명주 교수님의 사례. 매 번 라이더의 선택 데이터에 대해 LP를 정의하고 그 문제를 풀 것.
    # 보조금 부문을 제외하고 단순하게 작동하는 것으로 시험해 볼 것.
    w_data = []
    v1 = round(random.uniform(0.8, 1.2), 1)
    v2 = round(1000 * random.uniform(0.8, 1.2), 1)
    init_coeff = [v1, v2]
    while env.now <= cool_time:
        #interval 동안의 라이더 선택을 관찰
        v_old, rider_names, cts_name, d_orders_res, times, end_times = SubsidyPolicy_class.ProblemInput(rider_set, customer_set, env.now)  # 문제의 입력 값을 계산
        yield env.timeout(interval)
        now = round(env.now, 1)
        #선택된 라이더들에 대한 가치 함수 갱신
        current_w_data = []
        for rider_name in rider_set:
            rider = rider_set[rider_name]
            if env.now - interval <= rider.w_data[-1][0] < env.now:
                data = rider.w_data[-1]
                #r = [customer_set[data[1]].]
                current_w_data.append(rider.w_data[-1])
        #w_data로 LP풀기
        for data in current_w_data:
            init_coeff = lpg.ReviseCoeff_MJ(init_coeff, data, w_data, customer_set)
        print('예상 {} -> 실제 {}'.format(init_coeff, rider_set[0].coeff))



        actual_choice = RiderChoiceWithInterval(rider_set, rider_names, now, interval=interval)
        # [라이더 이름, 선택한 고객]
        expectation, actual = Comapre2List(platform_expected, actual_choice)
        print("차이발생/T now:",now,"/예상한 선택", expectation , "/실제 선택", actual)
        if len(expectation) > 0:
            input("계산 확인.")
            for info in expectation:
                rider = rider_set[info[0]]
                selected, others = InputCalculate2(rider, customer_set)
                ChoiceCheck(rider, customer_set)
                past_choices = []
                indexs = list(range(len(rider.choice_info) - 1))
                indexs.reverse()
                for index1 in indexs:
                    past_select, past_others = InputCalculate2(rider, customer_set, index=index1)
                    if len(past_others) > 0:
                        past_choices.append(
                            [past_select, past_others])  # 비교 대상이 필요함. 만약 past_others = empty 라면, 비교 식을 선언할 수 없음.
                    print("추가되는 사례", len(past_choices))
                    print('past_choices', past_choices)
                    #input("추가 사례 확인")
                input("계산 확인2.")
                feasibility, res = lpg.ReviseCoeffAP(selected, others, rider.p_coeff, past_data=past_choices)
                # 계수 갱신
                if feasibility == True:
                    for index in range(len(res)):
                        rider.p_coeff[index] += res[index]
                    print("갱신됨::", res)
            pass
        # 보조금 초기화
        Basic.InitializeSubsidy(customer_set)  # 보조금 초기화
        Basic.DefreezeAgent(rider_set, type='rider')  # 라이더 반영
        Basic.DefreezeAgent(customer_set, type='customer')  # 고객 반영
        if checker == False:
            print('Sys Run/T:' + str(env.now))
        else:
            input('Sys Run/T:' + str(env.now))
        input("정보확인1")
"""
