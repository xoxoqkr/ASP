# -*- coding: utf-8 -*-
#.py 설명 :
# -> LinearizedASP_gurobi.py : gurobipy를 이용해 보조금 할당 문제를 풀이하는 함수

import gurobipy as gp
from gurobipy import GRB
import random
import subsidyASP as ASP
import numpy



def LinearizedSubsidyProblem(driver_set, customers_set, v_old, ro, times, end_times, fee_weights = [],lower_b = False, upper_b = False, sp=None, print_gurobi=False,  solver=-1, delta = 1, relax = 100, min_subsidy = 0,
                             slack_time = 0):
    """
    선형화된 버전의 보조금 문제
    :param driver_set: 가능한 라이더 수
    :param customers_set: 가능한 고객 수
    :param v_old: 가치
    :param ro: 라이더 선택 순서
    :param lower_b: 보조금 상한
    :param upper_b: 보조금 하한
    :param sp: 우선 적으로 서비스 되어야 하는 고객 <-C_p
    :param print_gurobi: Gurobi 실행 내용 print문
    :param solver: gurobi의 solver engine 선택 [-1,0,1,2]
    :return: 해
    """
    #print('fee_weights',fee_weights)
    drivers = list(range(len(driver_set)))
    customers = list(range(len(customers_set)))
    driver_num = len(driver_set)
    customer_num = len(customers_set)
    sum_i = sum(ro)
    #print('parameters',drivers,customers, ro, driver_num, customer_num, sum_i)
    if upper_b == False:
        upper_b = 10000
    if lower_b == False:
        lower_b = 0
    # D.V. and model set.
    m = gp.Model("mip1")
    x = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="x")
    v = m.addVars(len(drivers), len(customers), lb = min_subsidy, vtype=GRB.CONTINUOUS, name="v")
    cso = m.addVars(len(customers), vtype=GRB.INTEGER, name="c" )
    #선형화를 위한 변수
    y = m.addVars(len(drivers), len(customers), vtype=GRB.BINARY, name="y")
    w = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="w")
    z = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="z")
    b = m.addVars(len(drivers), len(customers), vtype=GRB.CONTINUOUS, name="b") #크시
    #우선 고객 할당.
    if sp == None:
        num_sp = max(int(len(drivers) / 2), random.choice(drivers))
        sp = random.sample(customers, num_sp)
    rev_sp = sp
    req_sp_num = min(len(driver_set), len(sp), relax)
    #print("Priority Customer", rev_sp)
    # Set objective #29
    m.setObjective(gp.quicksum(v[i, j]/fee_weights[i] for i in drivers for j in customers), GRB.MINIMIZE)
    #32
    """
    for i in drivers:
        for j in customers:
            m.addConstr(gp.quicksum(w[i, k] + v_old[i, k] * x[i, k] for k in customers) >= z[i, j] + v_old[i, j] * y[i, j])
    """
    m.addConstrs(x[i,j]*(v_old[i,j] + v[i,j]) >= 0  for i in drivers for j in customers) #todo: 선택한 주문에 대한 비음 제약식.
    m.addConstrs(gp.quicksum(w[i,k] + v_old[i,k]*x[i,k] for k in customers) >= z[i,j] + v_old[i,j]*y[i,j] + delta for i in drivers for j in customers)
    #33
    m.addConstrs( w[i,j]-v[i,j ]<= upper_b*(1-x[i,j]) for i in drivers for j in customers)
    #34
    m.addConstrs(v[i, j] - w[i, j] <= upper_b*(1 - x[i, j]) for i in drivers for j in customers)
    #35
    m.addConstrs(w[i, j] <= upper_b * x[i, j] for i in drivers for j in customers)
    #36
    m.addConstrs(z[i, j] - v[i, j] <= upper_b*(1 - y[i, j]) for i in drivers for j in customers)
    #37
    m.addConstrs(v[i, j] - z[i, j] <= upper_b * (1 - y[i, j]) for i in drivers for j in customers)
    #38
    m.addConstrs(z[i, j] <= upper_b * y[i, j] for i in drivers for j in customers)
    #39
    m.addConstrs(cso[j] >= ro[i]*y[i,j] for i in drivers for j in customers)
    #40
    m.addConstrs(cso[j] <= (ro[i])*(1- y[i,j]) + driver_num*y[i,j] for i in drivers for j in customers)
    #41
    #m.addConstrs(b[i,j] == ro[i] for i in drivers for j in customers)
    m.addConstrs(gp.quicksum(b[i, j] for j in customers) == ro[i] for i in drivers)
    #42
    m.addConstrs(b[i,j] - cso[j] <= driver_num*(1 - x[i,j]) for i in drivers for j in customers)
    #43
    m.addConstrs(cso[j] - b[i, j]<= driver_num*(1 - x[i, j]) for i in drivers for j in customers)
    #44
    m.addConstrs(b[i, j] <= (driver_num)*x[i,j] for i in drivers for j in customers)
    #45
    #m.addConstr(gp.quicksum(x[i, j] for i in drivers for j in rev_sp) >= req_sp_num)
    m.addConstr(gp.quicksum(x[i, j] for i in drivers for j in rev_sp) >= min(req_sp_num,len(driver_set)))
    print('RHS',min(req_sp_num,len(driver_set)))
    #46
    m.addConstrs(gp.quicksum(x[i, j] for j in customers) == 1 for i in drivers)
    #m.addConstrs(gp.quicksum(x[i, j] for j in customers) <= 1 for i in drivers)
    #47
    m.addConstrs(gp.quicksum(x[i, j] for i in drivers) <= 1 for j in customers)
    #49
    m.addConstr(gp.quicksum(cso[j] for j in customers) == sum_i + (driver_num)*(customer_num - driver_num))
    #50
    m.addConstrs(cso[j] <= driver_num for j in customers)
    #51 시간 제약식 -> 기존에 없던 제약식
    #m.addConstrs(x[i,j]*times[i,j] <= end_times[i,j] + slack_time  for i in drivers for j in customers) rev_1
    #m.addConstrs(x[i, j] * times[i, j] <= end_times[i, j]*(1 + abs(0.5*times[i,j]/end_times[i, j])) for i in drivers for j in customers) rev_2
    m.addConstrs(x[i, j] * times[i, j] <= end_times[i, j] for i in drivers for j in customers)
    """
    for i in drivers:
        for j in customers:
            try:
                add_t = 1 +(0.5*times[i,j]/end_times[i, j])
            except:
                add_t = 1
            m.addConstr(x[i, j] * times[i, j] <= end_times[i, j]*add_t)
    """
    for i in drivers:
        for j in customers:
            if lower_b != False:
                m.addConstr(lower_b <= v[i, j])
            if upper_b != False:
                if times[i,j] <= end_times[i,j]:
                    m.addConstr(v[i, j]/fee_weights[i] <= upper_b)
                else:
                    m.addConstr(v[i, j] / fee_weights[i] <= upper_b * 1.5)
    if print_gurobi == False:
        m.setParam(GRB.Param.OutputFlag, 0)
    m.Params.method = solver  # -1은 auto dedection이며, 1~5에 대한 차이.
    m.optimize()
    """
    res = ASP.printer(m.getVars(), [], len(drivers), len(customers))
    print('Obj val: %g' % m.objVal, "Solver", solver)
    c_list = []
    x_list = []
    y_list = []
    for val in m.getVars():
        if val.VarName[0] == 'c':
            c_list.append(int(val.x))
        elif val.VarName[0] == 'x':
            x_list.append(int(val.x))
        elif val.VarName[0] == 'y':
            y_list.append(int(val.x))
        else:
            pass
    print("CSO")
    print(c_list)
    c_list.sort()
    print(c_list)
    x_list = np.array(x_list)
    x_list = x_list.reshape(driver_num, customer_num)
    print("X")
    print(x_list)
    print("Y")
    y_list = np.array(y_list)
    y_list = y_list.reshape(driver_num, customer_num)
    print(y_list)
    """
    #print(m.getVars()[:10])
    try:
        print('Obj val: %g' % m.objVal, "Solver", solver)
        res = ASP.printer(m.getVars(), [], len(drivers), len(customers))
        return res, m.getVars()
    except:
        #print('Infeasible')
        #res = printer(m.getVars(), [], len(drivers), len(customers))
        return False, False


def ReviseCoeffAP(selected, others, org_coeff, past_data = [], error = 10):
    """
    라이더의 가치함수 갱신 문제
    -> 수정 중
    :param selected:
    :param others:
    :param org_coeff:
    :param past_data:
    :return:
    """
    coeff = list(range(len(org_coeff)))
    # D.V. and model set.
    m = gp.Model("mip1")
    x = m.addVars(len(selected), vtype=GRB.CONTINUOUS, name="x")
    z = m.addVars(1 + len(past_data), vtype = GRB.CONTINUOUS, name= "z")
    a = m.addVars(len(selected), vtype=GRB.CONTINUOUS, name="a")


    #m.setObjective(gp.quicksum(x[i] for i in coeff), GRB.MINIMIZE)

    m.setObjective(gp.quicksum(a[i] for i in coeff), GRB.MINIMIZE)
    m.addConstrs(a[i] == gp.abs_(x[i]) for i in coeff)

    z_count = 0
    #이번 selected와 other에 대한 문제 풀이
    m.addConstr(gp.quicksum((x[i] + org_coeff[i])*selected[i] for i in coeff) == z[z_count])
    m.addConstr(z[z_count] >= 0)
    for other_info in others:
        m.addConstr(gp.quicksum((x[i] + org_coeff[i])*other_info[i] for i in coeff) <= z[z_count] - error)
    z_count += 1
    #과거 정보를 적층하는 작업
    if len(past_data) > 0:
        for data in past_data:
            p_selected = data[0]
            p_others = data[1]
            print('p_selected',p_selected)
            print('p_others',p_others)
            m.addConstr(gp.quicksum((x[i] + org_coeff[i]) * p_selected[i] for i in coeff) == z[z_count])
            for p_other_info in p_others:
                print('p_other_info',p_other_info)
                m.addConstr(gp.quicksum((x[i] + org_coeff[i]) * p_other_info[i] for i in coeff) <= z[z_count] - error)
            z_count += 1
    #풀이
    m.optimize()
    try:
        print('Obj val: %g' % m.objVal)
        res = []
        for val in m.getVars():
            if val.VarName[0] == 'x':
                res.append(float(val.x))
        return True, res
    except:
        print('Infeasible')
        return False, None


def ReviseCoeffAP1_org(selected, others, org_coeff, past_data = [], Big_M = 1000, weight_sum = False):
    """
    라이더의 가치함수 갱신 문제 => LP1
    -> 수정 중
    :param selected: #선택된 주문 1개의 요소들 [x11,x12,x13]
    :param others: #선택되지 않은 주문들의 요소들 [[x11,x12,x13],[x21,x22,x23],...,[]]
    :param org_coeff:현재 coeff
    :param past_data: 과거 선택들
    :return:
    """
    weight_direction = [1,1,1]
    coeff_indexs = list(range(len(org_coeff)))
    dummy_indexs = list(range(1 + len(past_data)))
    max_data_size = [1 + len(others)]
    for info in past_data:
        max_data_size.append(1 + len(info[1]))
    max_data_size = max(max_data_size)
    error_term_indexs = list(range(max_data_size))
    # D.V. and model set.
    m = gp.Model("mip1")
    w = m.addVars(len(org_coeff), lb = 0, vtype=GRB.CONTINUOUS, name="w")
    y = m.addVars(1 + len(past_data), max_data_size, vtype = GRB.CONTINUOUS, name= "y") #error
    m.setObjective(gp.quicksum(y[i,j] for i in dummy_indexs for j in error_term_indexs), GRB.MINIMIZE)
    dummy_index = 0
    error_term_index = 0
    #계수 합 관련
    if weight_sum == True:
        m.addConstrs((w[i] + org_coeff[i] >= 0 for i in coeff_indexs), name = 'c3')
        m.addConstr((gp.quicksum((w[i] + org_coeff[i]) for i in coeff_indexs) == 3), name = 'c4')
    #이번 selected와 other에 대한 문제 풀이
    m.addConstr((gp.quicksum((w[i] + org_coeff[i])*selected[i]*weight_direction[i] for i in coeff_indexs) + y[dummy_index,error_term_index] >= 0), name = 'c5')
    error_term_index += 1
    Constr_count = 0
    for other_info in others:
        #print('compare',selected,other_info)
        m.addConstr(gp.quicksum((w[i] + org_coeff[i])*selected[i]*weight_direction[i] for i in coeff_indexs) + y[dummy_index,error_term_index]>=
                    gp.quicksum((w[j] + org_coeff[j])*other_info[j]*weight_direction[j] for j in coeff_indexs), name = 'c6-'+str(Constr_count))
        error_term_index += 1
        Constr_count += 1
    dummy_index += 1
    #과거 정보를 적층하는 작업
    if len(past_data) > 0:
        data_index = 0
        for data in past_data:
            p_selected = data[0]
            p_others = data[1]
            #print('p_selected',p_selected)
            #print('p_others',p_others)
            error_term_index = 0
            m.addConstr(gp.quicksum((w[i] + org_coeff[i]) * p_selected[i] * weight_direction[i] for i in coeff_indexs) + y[dummy_index,error_term_index] >= 0, name = 'c7-'+ str(data_index) )
            error_term_index += 1
            for p_other_info in p_others:
                m.addConstr(gp.quicksum((w[i] + org_coeff[i])*p_selected[i] * weight_direction[i] for i in coeff_indexs) + y[dummy_index,error_term_index] >=
                           gp.quicksum((w[j] + org_coeff[j])*p_other_info[j] * weight_direction[j] for j in coeff_indexs), name = 'c8-'+str(data_index) + '-'+str(error_term_index-1))
                error_term_index += 1
            dummy_index += 1
            data_index += 1
    #풀이
    m.Params.method = 2
    m.optimize()
    exe_t = m.Runtime
    #print(m.display())
    #print(m.getVars())
    m.write("test_file.lp")
    obj = []
    #input('모형 확인')
    try:
        print('Obj val: %g' % m.objVal)
        res = []
        for val in m.getVars():
            if val.VarName[0] == 'w':
                res.append(float(val.x))
            elif val.VarName[0] == 'y':
                obj.append(float(val.x))
            else:
                pass
        return True, res, exe_t, sum(obj)
    except:
        print('Infeasible')
        return False, None, exe_t, 0

def ReviseCoeffAP1(selected, others, org_coeff, past_data = [], Big_M = 1000, weight_sum = False):
    """
    라이더의 가치함수 갱신 문제 => LP1
    -> 수정 중
    :param selected: #선택된 주문 1개의 요소들 [x11,x12,x13]
    :param others: #선택되지 않은 주문들의 요소들 [[x11,x12,x13],[x21,x22,x23],...,[]]
    :param org_coeff:현재 coeff
    :param past_data: 과거 선택들
    :return:
    """
    weight_direction = [1,1,1]
    coeff_indexs = list(range(len(org_coeff)))
    dummy_indexs = list(range(1 + len(past_data)))
    max_data_size = [1 + len(others)]
    for info in past_data:
        max_data_size.append(1 + len(info[1]))
    max_data_size = max(max_data_size)
    error_term_indexs = list(range(max_data_size))
    # D.V. and model set.
    m = gp.Model("mip1")
    w = m.addVars(len(org_coeff), lb = 0, vtype=GRB.CONTINUOUS, name="w")
    y = m.addVars(1 + len(past_data), max_data_size, vtype = GRB.CONTINUOUS, name= "y") #error
    m.setObjective(Big_M*gp.quicksum(y[i,j] for i in dummy_indexs for j in error_term_indexs), GRB.MINIMIZE)
    dummy_index = 0
    error_term_index = 0
    #계수 합 관련
    if weight_sum == True:
        m.addConstrs((w[i]  >= 0 for i in coeff_indexs), name = 'c3')
        m.addConstr((gp.quicksum((w[i]) for i in coeff_indexs) == 1), name = 'c4')
    #이번 selected와 other에 대한 문제 풀이
    m.addConstr((gp.quicksum((w[i] )*selected[i]*weight_direction[i] for i in coeff_indexs) + y[dummy_index,error_term_index] >= 0), name = 'c5')
    error_term_index += 1
    Constr_count = 0
    for other_info in others:
        #print('compare',selected,other_info)
        m.addConstr(gp.quicksum((w[i])*selected[i]*weight_direction[i] for i in coeff_indexs) + y[dummy_index,error_term_index]>=
                    gp.quicksum((w[j])*other_info[j]*weight_direction[j] for j in coeff_indexs), name = 'c6-'+str(Constr_count))
        error_term_index += 1
        Constr_count += 1
    dummy_index += 1
    #과거 정보를 적층하는 작업
    if len(past_data) > 0:
        data_index = 0
        for data in past_data:
            p_selected = data[0]
            p_others = data[1]
            #print('p_selected',p_selected)
            #print('p_others',p_others)
            error_term_index = 0
            m.addConstr(gp.quicksum((w[i]) * p_selected[i] * weight_direction[i] for i in coeff_indexs) + y[dummy_index,error_term_index] >= 0, name = 'c7-'+ str(data_index) )
            error_term_index += 1
            for p_other_info in p_others:
                m.addConstr(gp.quicksum((w[i])*p_selected[i] * weight_direction[i] for i in coeff_indexs) + y[dummy_index,error_term_index] >=
                           gp.quicksum((w[j])*p_other_info[j] * weight_direction[j] for j in coeff_indexs), name = 'c8-'+str(data_index) + '-'+str(error_term_index-1))
                error_term_index += 1
            dummy_index += 1
            data_index += 1
    #풀이
    #m.Params.method = 2
    m.Params.Method = 2  # barrier
    m.Params.Crossover = 0  # no crossover
    m.optimize()
    exe_t = m.Runtime
    #print(m.display())
    #print(m.getVars())
    m.write("test_file.lp")
    obj = []
    #input('모형 확인')
    try:
        print('Obj val: %g' % m.objVal)
        res = []
        for val in m.getVars():
            if val.VarName[0] == 'w':
                res.append(float(val.x))
            elif val.VarName[0] == 'y':
                obj.append(float(val.x))
            else:
                pass
        return True, res, exe_t, sum(obj)
    except:
        print('Infeasible')
        return False, None, exe_t, 0



def ReviseCoeffAP2(selected, others, org_coeff, past_data = [], Big_M = 1000, weight_sum = False):
    """
    라이더의 가치함수 갱신 문제=> LP2
    :param selected: #선택된 주문 1개의 요소들 [x11,x12,x13]
    :param others: #선택되지 않은 주문들의 요소들 [[x11,x12,x13],[x21,x22,x23],...,[]]
    :param org_coeff:현재 coeff
    :param past_data: 과거 선택들
    :return:
    """
    weight_direction = [1,1,1]
    coeff_indexs = list(range(len(org_coeff)))
    dummy_indexs = list(range(1 + len(past_data)))
    max_data_size = [1 + len(others)]
    for info in past_data:
        max_data_size.append(1 + len(info[1]))
    max_data_size = max(max_data_size)
    error_term_indexs = list(range(max_data_size))
    # D.V. and model set.
    m = gp.Model("mip1")
    w = m.addVars(len(org_coeff), lb = -2, vtype=GRB.CONTINUOUS, name="w")
    y = m.addVars(1 + len(past_data), max_data_size, vtype = GRB.CONTINUOUS, name= "y")
    z = m.addVars(len(org_coeff), vtype=GRB.CONTINUOUS, name="z")
    #print(y)
    #input('확인')
    #m.setObjective(gp.quicksum(abs_(z[i]) for i in coeff_indexs) + Big_M * gp.quicksum(
    #    y[i, j] for i in dummy_indexs for j in error_term_indexs), GRB.MINIMIZE)
    m.setObjective(gp.quicksum(z[i] for i in coeff_indexs) + Big_M*gp.quicksum(y[i,j] for i in dummy_indexs for j in error_term_indexs), GRB.MINIMIZE)
    #m.addConstrs((y[i,j] == 0 for i in dummy_indexs for j in error_term_indexs), name='c0-dummy')
    m.addConstrs(( z[i]  - w[i]>= 0 for i in coeff_indexs), name= 'c1') #linearization part
    m.addConstrs(( z[i] +w[i] >= 0 for i in coeff_indexs), name = 'c2')
    #m.addConstrs(( z[i]  >= w[i] for i in coeff_indexs), name= 'c1') #linearization part
    #m.addConstrs(( z[i] >= -w[i] for i in coeff_indexs), name = 'c2')
    #m.addConstrs(org_coeff[i] - w[i] <= z[i] for i in coeff_indexs) #linearization part
    #m.addConstrs(org_coeff[i] - w[i]  >= -z[i] for i in coeff_indexs)
    dummy_index = 0
    error_term_index = 0
    #계수 합 관련
    if weight_sum == True:
        m.addConstrs((w[i] + org_coeff[i] >= 0 for i in coeff_indexs), name = 'c3')
        m.addConstr((gp.quicksum((w[i] + org_coeff[i]) for i in coeff_indexs) == 1), name = 'c4')
    #이번 selected와 other에 대한 문제 풀이
    m.addConstr((gp.quicksum((w[i] + org_coeff[i])*selected[i]*weight_direction[i] for i in coeff_indexs) + y[dummy_index,error_term_index] >= 0), name = 'c5')
    error_term_index += 1
    Constr_count = 0
    for other_info in others:
        #print('compare',selected,other_info)
        m.addConstr(gp.quicksum((w[i] + org_coeff[i])*selected[i]*weight_direction[i] for i in coeff_indexs) + y[dummy_index,error_term_index]>=
                    gp.quicksum((w[j] + org_coeff[j])*other_info[j]*weight_direction[j] for j in coeff_indexs), name = 'c6-'+str(Constr_count))
        error_term_index += 1
        Constr_count += 1
    dummy_index += 1
    #과거 정보를 적층하는 작업
    if len(past_data) > 0:
        data_index = 0
        for data in past_data:
            p_selected = data[0]
            p_others = data[1]
            #print('p_selected',p_selected)
            #print('p_others',p_others)
            error_term_index = 0
            m.addConstr(gp.quicksum((w[i] + org_coeff[i]) * p_selected[i] * weight_direction[i] for i in coeff_indexs) + y[dummy_index,error_term_index] >= 0, name = 'c7-'+ str(data_index) )
            error_term_index += 1
            for p_other_info in p_others:
                #print('p_other_info',p_other_info)
                m.addConstr(gp.quicksum((w[i] + org_coeff[i])*p_selected[i] * weight_direction[i] for i in coeff_indexs) + y[dummy_index,error_term_index] >=
                           gp.quicksum((w[j] + org_coeff[j])*p_other_info[j] * weight_direction[j] for j in coeff_indexs), name = 'c8-'+str(data_index) + '-'+str(error_term_index-1))
                error_term_index += 1
            dummy_index += 1
            data_index += 1
    #풀이
    m.Params.method = 2
    m.optimize()
    exe_t = m.Runtime
    #print(m.display())
    #print(m.getVars())
    m.write("test_file.lp")
    obj = []
    #input('모형 확인')
    try:
        print('Obj val: %g' % m.objVal)
        #print(w)
        #print(z)
        res = []
        for val in m.getVars():
            if val.VarName[0] == 'w':
                res.append(float(val.x))
            elif val.VarName[0] == 'y':
                obj.append(float(val.x))
            else:
                pass
        return True, res, exe_t, sum(obj)
    except:
        print('Infeasible')
        return False, None, exe_t, 0


def ReviseCoeffAP3(selected, others, org_coeff, past_data = [], Big_M = 100000, weight_sum = True, selected_nonnegative = True):
    """
    라이더의 가치함수 갱신 문제=> LP2
    :param selected: #선택된 주문 1개의 요소들 [x11,x12,x13]
    :param others: #선택되지 않은 주문들의 요소들 [[x11,x12,x13],[x21,x22,x23],...,[]]
    :param org_coeff:현재 coeff
    :param past_data: 과거 선택들
    :return:
    """
    print('LP3 coeff',org_coeff)
    weight_direction = [1,1,1]
    coeff_indexs = list(range(len(org_coeff)))
    dummy_indexs = list(range(1 + len(past_data)))
    max_data_size_infos = [1 + len(others)]
    for info in past_data:
        max_data_size_infos.append(1 + len(info[1]))
    max_data_size = max(max_data_size_infos)
    error_term_indexs = list(range(max_data_size))
    zero_y_indexs = [] #정의되지 않는 y에 대해 0 값을 만들기
    info_count = 0
    for info in past_data:
        size_diff = max_data_size - (1 + len(info[1]))
        if size_diff > 0:
            for index in range((1 + len(info[1])),max_data_size):
                zero_y_indexs.append([info_count, index])
        info_count += 1
    # D.V. and model set.
    m = gp.Model("mip1")
    w = m.addVars(len(org_coeff), lb = 0, vtype=GRB.CONTINUOUS, name="w")
    y = m.addVar(lb = 0, ub=Big_M, vtype=GRB.CONTINUOUS, name="y")
    #y = m.addVars(1 + len(past_data), max_data_size, ub = Big_M, vtype = GRB.CONTINUOUS, name= "y")
    #Objective Function
    m.setObjective(y, GRB.MAXIMIZE)
    #m.setObjective(gp.quicksum(y[i,j] for i in dummy_indexs for j in error_term_indexs) , GRB.MAXIMIZE)
    dummy_index = 0
    error_term_index = 0
    #계수 합 관련
    #m.addConstr(w[1] == 0) #todo : 2차원만 고려해 보자.
    #m.addConstr(y == 0)
    if weight_sum == True:
        m.addConstrs((w[i] >= 0 for i in coeff_indexs), name = 'c3')
        m.addConstr((gp.quicksum((w[i] ) for i in coeff_indexs) == 1), name = 'c4')
    #이번 selected와 other에 대한 문제 풀이
    if selected_nonnegative == True:
        m.addConstr((gp.quicksum((w[i])*selected[i]*weight_direction[i] for i in coeff_indexs) >= 0), name = 'c5')
    error_term_index += 1
    Constr_count = 0
    for other_info in others:
        #print('compare',selected,other_info)
        m.addConstr(gp.quicksum((w[i])*selected[i]*weight_direction[i] for i in coeff_indexs) >=
                    gp.quicksum((w[j])*other_info[j]*weight_direction[j] for j in coeff_indexs) + y , name = 'c6-'+str(Constr_count))
        #print('자료',list(org_coeff),list(selected[:3]),list(other_info[:3]))
        tem1 = numpy.dot(org_coeff,list(selected[:3]))
        tem2 = numpy.dot(org_coeff,list(other_info[:3]))
        if abs(tem2-tem1) > 1:
            print('LP3 비교_1',tem1,'>=',tem2, '차이', tem2-tem1, '고객',selected[3],'vs',other_info[3])
        error_term_index += 1
        Constr_count += 1
    dummy_index += 1
    #과거 정보를 적층하는 작업
    if len(past_data) > 0:
        data_index = 0
        for data in past_data:
            p_selected = data[0]
            p_others = data[1]
            #print('p_selected',p_selected)
            #print('p_others',p_others)
            error_term_index = 0
            if selected_nonnegative == True:
                m.addConstr(gp.quicksum((w[i]) * p_selected[i] * weight_direction[i] for i in coeff_indexs)  >= 0, name = 'c7-'+ str(data_index) )
            error_term_index += 1
            for p_other_info in p_others:
                #print('p_other_info',p_other_info)
                m.addConstr(gp.quicksum((w[i])*p_selected[i] * weight_direction[i] for i in coeff_indexs) >=
                           gp.quicksum((w[j])*p_other_info[j] * weight_direction[j] for j in coeff_indexs)+ y, name = 'c8-'+str(data_index) + '-'+str(error_term_index-1))
                #print('자료', list(org_coeff), list(p_selected[:3]), list(p_other_info[:3]))
                tem1 = numpy.dot(org_coeff, list(p_selected[:3]))
                tem2 = numpy.dot(org_coeff, list(p_other_info[:3]))
                if abs(tem2 - tem1) > 1:
                    print('LP3 비교2', tem1, '>=', tem2, '차이', tem2 - tem1)
                error_term_index += 1
            dummy_index += 1
            data_index += 1
    #제약식이 존재하지 않는 zero_y_indexs의 값을 0으로 하기.
    #for zero_index in zero_y_indexs:
    #    m.addConstr(y[zero_index[0], zero_index[1]] == 0)
    #풀이
    m.Params.method = 2
    m.optimize()
    exe_t = m.Runtime
    #print(m.display())
    #print(m.getVars())
    ite = str(len(past_data))
    #m.write("LP3_file"+ite+".lp")
    obj = []
    revise_obj = []
    """
    #input('모형 확인')
    #Binding Constraint 보기
    check2 = []
    for val in m.getVars():
        if val.VarName[0] == 'w':
            check2.append(float(val.x))
        elif val.VarName[0] == 'y':
            check2.append(float(val.x))
    print('LP3 체크',check2) 
    if 0 in check2[:3]:
        print(check2)
        #input('쏠림 발생')       
    """
    try:
        res = []
        for val in m.getVars():
            if val.VarName[0] == 'w':
                res.append(float(val.x))
            elif val.VarName[0] == 'y':
                obj.append(float(val.x))
                if int(val.x) != Big_M:
                    revise_obj.append(val.x)
            else:
                pass
        #if m.objVal < 0:
        #    input('LP3 eta 음수', res)
        return True, res, exe_t, sum(revise_obj)
    except:
        res = []
        for val in m.getVars():
            if val.VarName[0] == 'w':
                try:
                    res.append(float(val.x))
                except:
                    pass
        #print('Obj val: %g' % m.objVal)
        #print('Infeasible', res)
        return False, None, 0, 0


def ReviseCoeffAP3_WIP(selected, others, org_coeff, past_data = [], Big_M = 100000, weight_sum = False):
    """
    라이더의 가치함수 갱신 문제=> LP2
    :param selected: #선택된 주문 1개의 요소들 [x11,x12,x13]
    :param others: #선택되지 않은 주문들의 요소들 [[x11,x12,x13],[x21,x22,x23],...,[]]
    :param org_coeff:현재 coeff
    :param past_data: 과거 선택들
    :return:
    """
    weight_direction = [1,1,1]
    coeff_indexs = list(range(len(org_coeff)))
    dummy_indexs = list(range(1 + len(past_data)))
    max_data_size_infos = [1 + len(others)]
    for info in past_data:
        max_data_size_infos.append(1 + len(info[1]))
    max_data_size = max(max_data_size_infos)
    error_term_indexs = list(range(max_data_size))
    zero_y_indexs = [] #정의되지 않는 y에 대해 0 값을 만들기
    info_count = 0
    for info in past_data:
        size_diff = max_data_size - (1 + len(info[1]))
        if size_diff > 0:
            for index in range((1 + len(info[1])),max_data_size):
                zero_y_indexs.append([info_count, index])
        info_count += 1
    # D.V. and model set.
    m = gp.Model("mip1")
    w = m.addVars(len(org_coeff), lb = -2, vtype=GRB.CONTINUOUS, name="w")
    y = m.addVar(ub = Big_M, vtype = GRB.CONTINUOUS, name= "y")
    #Objective Function
    m.setObjective(y , GRB.MAXIMIZE)
    dummy_index = 0
    error_term_index = 0
    #계수 합 관련
    if weight_sum == True:
        m.addConstrs((w[i] >= 0 for i in coeff_indexs), name = 'c3')
        m.addConstr((gp.quicksum((w[i] ) for i in coeff_indexs) == 1), name = 'c4')
    #이번 selected와 other에 대한 문제 풀이
    m.addConstr((gp.quicksum((w[i])*selected[i]*weight_direction[i] for i in coeff_indexs)  >= 0), name = 'c5')
    error_term_index += 1
    Constr_count = 0
    for other_info in others:
        #print('compare',selected,other_info)
        m.addConstr(gp.quicksum((w[i])*selected[i]*weight_direction[i] for i in coeff_indexs) >=
                    gp.quicksum((w[j])*other_info[j]*weight_direction[j] for j in coeff_indexs) + y, name = 'c6-'+str(Constr_count))
        error_term_index += 1
        Constr_count += 1
    dummy_index += 1
    #과거 정보를 적층하는 작업
    if len(past_data) > 0:
        data_index = 0
        for data in past_data:
            p_selected = data[0]
            p_others = data[1]
            #print('p_selected',p_selected)
            #print('p_others',p_others)
            error_term_index = 0
            m.addConstr(gp.quicksum((w[i]) * p_selected[i] * weight_direction[i] for i in coeff_indexs)  >= 0, name = 'c7-'+ str(data_index) )
            error_term_index += 1
            for p_other_info in p_others:
                #print('p_other_info',p_other_info)
                m.addConstr(gp.quicksum((w[i])*p_selected[i] * weight_direction[i] for i in coeff_indexs) >=
                           gp.quicksum((w[j])*p_other_info[j] * weight_direction[j] for j in coeff_indexs) + y, name = 'c8-'+str(data_index) + '-'+str(error_term_index-1))
                error_term_index += 1
            dummy_index += 1
            data_index += 1
    #제약식이 존재하지 않는 zero_y_indexs의 값을 0으로 하기.
    #for zero_index in zero_y_indexs:
    #    m.addConstr(y[zero_index[0], zero_index[1]] == 0)
    #풀이
    m.Params.method = 2
    m.optimize()
    exe_t = m.Runtime
    #print(m.display())
    #print(m.getVars())
    m.write("test_file.lp")
    obj = []
    revise_obj = []
    #input('모형 확인')
    try:
        print('Obj val: %g' % m.objVal)
        #input('LP3확인')
        #print(w)
        #print(z)
        res = []
        for val in m.getVars():
            if val.VarName[0] == 'w':
                res.append(float(val.x))
            elif val.VarName[0] == 'y':
                obj.append(float(val.x))
                if int(val.x) != Big_M:
                    revise_obj.append(val.x)
            else:
                pass
        return True, res, exe_t, sum(revise_obj)
    except:
        print('Infeasible')
        return False, None, exe_t, 0


def LogScore(coeff, vector):
    score = 0
    #print('coeff {} vector {} '.format(coeff, vector))
    for index in range(len(coeff)):
        try:
            score += (coeff[index] ** vector[index])*((-1)**index)
        except:
            input('스칼라 오류 coeff :{}, vector : {}'.format(coeff, vector))
    #print('지수 점수 {}'.format(score) )
    return round(score,4)


def ValueCal(coeff, vector, cal_type = 'linear'):
    if cal_type == 'log':
        val = LogScore(coeff, vector)
    elif cal_type == 'linear':
        val = numpy.dot(coeff, vector)
    else:
        val = 0
        print("Error")
    return round(val,2)


def UpdateGurobiModel(coeff, data, past_data = [], print_para = False, cal_type = 'linear', M = 1000):
    """
    주어진 coeff를 새로운 data에 대해 재계산하고 갱신함.
    Args:
        coeff: weight vector
        data: new data [[selected order], [other 1],...,[other n]]
        past_data: optional [data0, data1,...,data n]
        print_para: print option
        cal_type: print option 2
        M: penalty for slack variable y

    Returns: weight vector

    """
    #확장형 1.2를 고려한 확장형2
    w_index = list(range(len(coeff)))
    # D.V. and model set.
    model = gp.Model("problem1_3")
    w = model.addVars(len(coeff), vtype=GRB.CONTINUOUS, name="w")
    z = model.addVars(len(coeff), vtype=GRB.CONTINUOUS, name="z")
    y = model.addVars(len(data) + len(past_data), vtype=GRB.CONTINUOUS, name="y")

    model.setObjective(gp.quicksum(z) + M*gp.quicksum(y), GRB.MINIMIZE)

    model.addConstrs(coeff[i] - w[i] <= z[i] for i in w_index) #linearization part
    model.addConstrs(coeff[i] - w[i]  >= -z[i] for i in w_index)

    z_count = 0 #D_new part
    if print_para == True:
        score = ValueCal(coeff, data[0], cal_type=cal_type)
        print('선택 고객 z {} '.format(score))
    #print('확인 {} : {}'.format(coeff, data[0]))
    z_val = ValueCal(coeff, data[0], cal_type=cal_type)
    data_index = 0
    for other_info in data[1:]:
        compare_val = ValueCal(coeff, other_info, cal_type=cal_type)
        if print_para == True:
            if z_val < compare_val:
                print('Current {}-{} 비교 결과 Z : {} < {} : Val'.format(z_count, data_index, z_val, compare_val))
            else:
                print('Current {}-{} 비교 결과 Z : {} > {} : Val'.format(z_count, data_index, z_val, compare_val))
        model.addConstr(gp.quicksum(data[0][i]*(coeff[i] + w[i]) for i in w_index) + y[z_count]
                    >= gp.quicksum(data[data_index][i]*(coeff[i] + w[i]) for i in w_index))
        data_index += 1
    z_count += 1
    #2 model 수정
    if len(past_data) > 0:
        for data in past_data:
            z_val_old = ValueCal(coeff, data[0], cal_type=cal_type)
            p_selected = data[0]
            p_others = data[1:]
            data_index2 = 0
            for p_other_info in p_others:
                compare_val_old = ValueCal(coeff, p_other_info, cal_type=cal_type)
                if print_para == True:
                    if z_val_old < compare_val_old:
                        print('Past {}-{} 비교 결과 Z : {} < {} : Val'.format(z_count, data_index2,  z_val_old, compare_val_old))
                    else:
                        print('Past {}-{} 비교 결과 Z : {} > {} : Val'.format(z_count, data_index2, z_val_old, compare_val_old))
                    pass
                model.addConstr(gp.quicksum(p_selected[i] * (coeff[i] + w[i]) for i in w_index) + y[z_count]
                                >= gp.quicksum(p_other_info[i] * (coeff[i] + w[i]) for i in w_index))
                data_index2 += 1
            z_count += 1
    #3 model 풀이
    model.setParam(GRB.Param.OutputFlag, 0)
    model.Params.method = -1
    model.optimize()
    try:
        #print('Obj val 출력: %g' % model.objVal)
        res = []
        for val in model.getVars():
            if val.VarName[0] == 'w':
                res.append(float(val.x))
        print('결과 {} '.format(res))
        return True, res, model
    except:
        print('Infeasible')
        return False, None, model


def ReviseCoeff_MJ(coeff, now_data, past_data, error = 10):
    coeff = list(range(len(coeff)))
    # D.V. and model set.
    m = gp.Model("mip1")
    x = m.addVars(len(coeff), vtype=GRB.CONTINUOUS, name="x")
    z = m.addVars(1 + len(past_data), vtype = GRB.CONTINUOUS, name= "z")
    a = m.addVars(len(coeff), vtype=GRB.CONTINUOUS, name="a")


    #m.setObjective(gp.quicksum(x[i] for i in coeff), GRB.MINIMIZE)

    m.setObjective(gp.quicksum(a[i] for i in coeff), GRB.MINIMIZE)
    m.addConstrs(a[i] == gp.abs_(x[i]) for i in coeff)
    z_count = 0
    #이번 selected와 other에 대한 문제 풀이
    m.addConstr(gp.quicksum((x[i] + coeff[i])*now_data[0][i] for i in coeff) == z[z_count])
    m.addConstr(z[z_count] >= 0)
    for other_info in now_data[1]:
        m.addConstr(gp.quicksum((x[i] + coeff[i])*other_info[i] for i in coeff) <= z[z_count] - error)
    z_count += 1
    #과거 정보를 적층하는 작업
    if len(past_data) > 0:
        for data in past_data:
            p_selected = data[0]
            p_others = data[1]
            m.addConstr(gp.quicksum((x[i] + coeff[i]) * p_selected[i] for i in coeff) == z[z_count])
            for p_other_info in p_others:
                m.addConstr(gp.quicksum((x[i] + coeff[i]) * p_other_info[i] for i in coeff) <= z[z_count] - error)
            z_count += 1
    #풀이
    m.optimize()
    try:
        print('Obj val: %g' % m.objVal)
        res = []
        for val in m.getVars():
            if val.VarName[0] == 'x':
                res.append(float(val.x))
        return True, res
    except:
        print('Infeasible')
        return False, None
"""
#LinearizedSubsidyProblem(driver_set, customers_set, v_old, ro, lower_b = False, upper_b = False, sp=None, print_gurobi=False,  solver=-1)
driver_num = 10
customer_num = 20
driver_set = list(range(1,driver_num + 1))
customers_set = list(range(1,customer_num + 1))
orders = list(range(1,driver_num + 1))
#orders = list(range(driver_num))
print('drivers',orders)
random.shuffle(orders)
v_old = []
for i in range(driver_num*customer_num):
    v_old.append(random.randrange(5,25))
v_old = np.array(v_old)
v_old = v_old.reshape(driver_num,customer_num)
print(np.shape(v_old))
times = np.zeros((driver_num,customer_num))
end_times = np.zeros((driver_num,customer_num))
#res , vars = LinearizedSubsidyProblem(driver_set, customers_set, v_old, orders,times, end_times, print_gurobi = True)
"""