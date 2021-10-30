# -*- coding: utf-8 -*-
import gurobipy as gp
from gurobipy import GRB
import random
import numpy as np
import operator
import math
import copy

#1명의 라이더에 대해서 자료를 생성
class Rider(object):
    def __init__(self, name,coeff, p_coeff):
        self.name = name
        self.coeff = coeff
        self.p_coeff = p_coeff
        self.exp_pool = []
        self.actual_pool = []
        self.last_loc = None
        self.old_coeff = copy.deepcopy(p_coeff)

class Customer(object):
    def __init__(self, name,u,t):
        self.name = name
        self.u = u
        self.gen_t = t

def CustomerGen2(num, t, wrong_pr = 0.5 ,speed = 130, basic = 3000, wagePerhour = 9000, p_coeff = [1,1,1], r_coeff = [1,1,1]):
    """
    고객 생성 함수
    :param num: 고객수
    :param t: data 인덱스
    :param speed: 차량 속도
    :param basic: 기본료
    :param wagePerhour: 라이더 시간당 임금
    :return:
    """
    res = []
    values = []
    for count in range(num - 1):
        #1 고객들의 좌표를 생성
        store_loc = [random.randrange(0,50) , random.randrange(0,50)] #눈금 한칸이 100미터임!
        customer_loc = [random.randrange(0,50) , random.randrange(0,50)]
        #2 좌표에 대한 수수료를 생성
        dist = round(math.sqrt((store_loc[0] - customer_loc[0])**2 + (store_loc[1] - customer_loc[1])**2),4)*100
        fee = basic + 100*(max(0, dist//100 - 10))
        cost = (dist/speed)*(wagePerhour/60)
        order_type = random.randrange(1,3)
        info = [fee, cost, order_type]
        #print("확인1",int(dist), speed, fee, round(cost), order_type, store_loc, customer_loc)
        c = Customer(count, info, t)
        res.append(c)
        values.append(ValueCal(r_coeff, info))
    count += 1
    count1 = 0
    wr_pass = False
    max_r_value = max(values)
    while count1 < 10000: #마지막 고객에대해 예상과 다른 고객이 생성되도록 유도
        # 1 고객들의 좌표를 생성
        store_loc = [random.randrange(0, 50), random.randrange(0, 50)]
        customer_loc = [random.randrange(0, 50), random.randrange(0, 50)]
        # 2 좌표에 대한 수수료를 생성
        dist = round(math.sqrt((store_loc[0] - customer_loc[0])**2 + (store_loc[1] - customer_loc[1])**2),4)*100
        fee = basic + 100*(max(0, dist//100 - 10))
        cost = (dist/speed)*(wagePerhour/60)
        order_type = random.randrange(1,3)
        info = [fee, cost, order_type]
        p_value = ValueCal(p_coeff, info)
        r_value = ValueCal(r_coeff, info)
        if p_value < r_value and r_value < max_r_value:
            wr_pass = True
            c = Customer(count, info, t)
            res.append(c)
            print("다른 고객 발생함.")
            break
        count1 += 1
    if wr_pass == False:
        # 1 고객들의 좌표를 생성
        store_loc = [random.randrange(0, 50), random.randrange(0, 50)]
        customer_loc = [random.randrange(0, 50), random.randrange(0, 50)]
        # 2 좌표에 대한 수수료를 생성
        dist = round(math.sqrt((store_loc[0] - customer_loc[0])**2 + (store_loc[1] - customer_loc[1])**2),4)*100
        fee = basic + 100*(max(0, dist//100 - 10))
        cost = (dist/speed)*(wagePerhour/60)
        order_type = random.randrange(1,3)
        info = [fee, cost, order_type]
        c = Customer(count, info, t)
        res.append(c)
    return res


def ValueCal(coeff, data):
    """
    coeff*data 연산 수행
    :param coeff:
    :param data:
    :return:
    """
    index = 0
    value = 0
    #print('coeff:', coeff)
    #print('data :', data)
    for i in coeff:
        added = i*data[index]
        #print(added,)
        value += added
        index += 1
    return round(value,4)


def RandomData2(para1s, para2s, size, type = 'normal', data  = None, positive = True):
    """
    :param para1:
    :param para2:
    :param size: 분포의 population 크기
    :param num: 고객의 수
    :param num_coeff: 고객의 요소 수
    :param type:
    :param data:
    :return:
    """
    values = []
    for index in range(len(para1s)):
        para1 = para1s[index]
        para2 = para2s[index]
        if type == 'normal':
            pool = np.random.normal(para1, para2, size=size)
        elif type == 'uniform':
            pool = np.random.uniform(low=para1, high=para2, size=size)
        elif type == 't':
            pool = np.random.standard_t(df=para1, size=size)
        elif type == 'F':# F븐포
            pool = np.random.f(dfnum=para1, dfden=para2, size=size)
        else: #주어진 데이터 값이 있는 경우
            pool = data
        #sample_pool 생성
        if positive == True:
            sample = None
            ite = 0
            while sample == None or ite < 1000:
                value = random.choice(list(pool))
                if value > 0:
                    sample = value
                    break
        else:
            sample = random.choice(list(pool))
        values.append(round(sample, 4))
    return values



def Choice(coeff, customers, selected = [], minus_para = False):
    """
    입력 받은 customers 중 가장 큰 가치 값을 가지는 고객을 선택
    :param coeff: 라이더의 가치함수
    :param customers: 고객들
    :param selected: 이미 선택된 고객들
    :param minus_para: Fasle 계산된 value에 상관 없이 가장 큰 값을 선택/ True : value가 양수인 경우에만 값을 반환
    :return:
    """
    cnadidates = []
    scores = []
    rank = []
    for customer in customers:
        if customer.name not in selected:
            value = customer.u
            score = round(ValueCal(coeff, value),4)
            scores.append(score)
            cnadidates.append([customer.name, score, value])
            #print('고객',customer.name,"총 점수",score,'개별 점수',)

    choice_value = max(scores)
    for score in sorted(scores, reverse= True):
        name = scores.index(score)
        rank.append(cnadidates[name][0])
    if minus_para == True:
        if choice_value > 0:
            index = scores.index(choice_value)
            cnadidates.sort(key = operator.itemgetter(1))
            return choice_value, customers[index].name, cnadidates, rank
        else:
            return 0, None, None, None
    else:
        index = scores.index(choice_value)
        cnadidates.sort(key = operator.itemgetter(1))
        return choice_value, customers[index].name, cnadidates, rank


def SingleRun(rider, customers, type = 'platform', min_value = None):
    res = []
    selected = []
    choices = []
    if type == 'platform':
        #print("플랫폼 계산")
        choice, select, pool , rank = Choice(rider.p_coeff, customers, selected)
        rider.exp_pool.append(pool)
    else:
        choice, select, pool, rank  = Choice(rider.coeff, customers, selected)
        rider.actual_pool.append(pool)
    rider.exp_pool = pool
    if min_value == None:
        res.append([rider.name, select]) #[[라이더 이름, 고객 이름],...,]
        selected.append(select)
    else: #min value에 값이 있는 경우(라이더는 이 값 이하인 주문을 수락하지 않음)
        if choice >= min_value:
            res.append([rider.name, select])  # [[라이더 이름, 고객 이름],...,]
            selected.append(select)
            choices.append(choice)
        else:
            res.append([rider.name, None])  # [[라이더 이름, 고객 이름],...,]
            selected.append(None)
            choices.append(choice)
    return res[0], selected[0], choices[0], rank


def CoeffRevise(selected, expect, others, org_coeff,  lbs, ubs, ite = 1, print_gurobi = False, constraint_para = False):
    coeff = list(range(len(org_coeff)))
    num_others = list(range(len(others)))
    # D.V. and model set.
    m = gp.Model("mip1")
    x = m.addVars(len(org_coeff), vtype=GRB.CONTINUOUS, name="x")
    l2 = m.addVars(len(org_coeff), vtype=GRB.CONTINUOUS, name="l2")
    sum_l = m.addVar(vtype = GRB.CONTINUOUS, name = "_sum_l")
    sum_x = m.addVar(vtype = GRB.CONTINUOUS, name = "_sum_x")
    #목적식
    m.setObjective(1/2*sum_x + ite*sum_l, GRB.MINIMIZE)
    m.addConstr(sum_x == gp.quicksum(x[i]*x[i] for i in coeff))
    for i in coeff:
        m.addConstr(l2[i] == gp.quicksum((x[i] + org_coeff[i])*selected[i] - (x[i] + org_coeff[i])*expect[i] for i in coeff))
    m.addConstr(sum_l == gp.quicksum(l2[i]*l2[i] for i in coeff))

    #2 a1,a2,a3 하한과 상항
    m.addConstrs(org_coeff[i] + x[i] >= lbs[i] for i in coeff)
    m.addConstrs(org_coeff[i] + x[i] <= ubs[i] for i in coeff)

    if print_gurobi == False:
        m.setParam(GRB.Param.OutputFlag, 0)
    #풀이
    m.Params.method = 0
    m.params.NonConvex = 2
    m.optimize()
    try:
        print('Obj val: %g' % m.objVal)
        res = []
        for val in m.getVars():
            #print('VarName', val.VarName)
            if val.VarName[0] == 'x':
                res.append(float(val.x))
        return True, res
    except:
        #print('Infeasible')
        return False, None


def CoeffRevise2(selected, expect, others, org_coeff, lbs, ubs, ite = 1, print_gurobi = False):
    coeff = list(range(len(org_coeff)))
    num_others = list(range(len(others)))
    # D.V. and model set.
    m = gp.Model("value_revise")
    x = m.addVars(len(org_coeff), vtype=GRB.CONTINUOUS, name="x")
    l2 = m.addVars(len(org_coeff), vtype=GRB.CONTINUOUS, name="l2")
    sum_l = m.addVar(vtype = GRB.CONTINUOUS, name = "_sum_l")
    sum_x = m.addVar(vtype = GRB.CONTINUOUS, name = "_sum_x")
    abs_l = m.addVar(vtype = GRB.CONTINUOUS, name = "_abs_x")
    #목적식
    #m.setObjective(gp.quicksum(x[i] for i in coeff)+ ite*abs_l, GRB.MINIMIZE)
    m.setObjective(1/2*sum_x + ite*abs_l, GRB.MINIMIZE)
    m.addConstr(sum_x == gp.quicksum(x[i]*x[i] for i in coeff))
    for i in coeff:
        m.addConstr(l2[i] == gp.quicksum((x[i] + org_coeff[i])*selected[i] - (x[i] + org_coeff[i])*expect[i] for i in coeff))
        pass
    #m.addConstr(sum_l == gp.quicksum(l2[i] for i in coeff))
    #m.addConstr(abs_l == gp.abs_s(sum_l))
    #1 1-term
    #2 2-term
    #2 a1,a2,a3 하한과 상항
    m.addConstrs(org_coeff[i] + x[i] >= lbs[i] for i in coeff)
    m.addConstrs(org_coeff[i] + x[i] <= ubs[i] for i in coeff)
    #m.addConstrs(x[i] <= -0.1 for i in coeff)
    if print_gurobi == False:
        m.setParam(GRB.Param.OutputFlag, 0)
    #풀이
    m.Params.method = 1
    #m.params.NonConvex = 2
    m.optimize()
    try:
        print('Obj val: %g' % m.objVal)
        res = []
        for val in m.getVars():
            #print('VarName', val.VarName)
            if val.VarName[0] == 'x':
                res.append(float(val.x))
        return True, res
    except:
        #print('Infeasible')
        return False, None


def CoeffRevise3(selected, org_coeff,  lbs, ubs , ite, past_data = [], print_gurobi = False, solver = 0):
    coeff = list(range(len(org_coeff)))
    # D.V. and model set.
    m = gp.Model("mip1")
    x = m.addVars(len(org_coeff),  vtype=GRB.CONTINUOUS, name="x")
    abs_x = m.addVars(len(org_coeff),  vtype=GRB.CONTINUOUS, name="abs_x")

    #l2 = m.addVars(len(org_coeff), vtype=GRB.CONTINUOUS, name="l2")
    #sum_l = m.addVar(vtype = GRB.CONTINUOUS, name = "_sum_l")
    #sum_x = m.addVar(vtype = GRB.CONTINUOUS, name = "_sum_x")
    #목적식
    #m.setObjective(gp.quicksum(x[i]*x[i] for i in coeff), GRB.MINIMIZE)
    m.setObjective(gp.quicksum(abs_x[i] for i in coeff), GRB.MINIMIZE)
    m.addConstrs(abs_x[i] == gp.abs_(x[i]) for i in coeff )
    #1 기존의 선택된 고객들이 유지되도록
    #1-1 선택된 주문의 가치 > 0
    m.addConstr(gp.quicksum(selected[i]*(org_coeff[i] + x[i]) for i in coeff) >= 0)
    #1-2 다른 주문과 비교하였을 때 선택된 주문의 가치가 가장 큼
    for info in past_data:
        p_select = info[0]
        p_others = info[1]
        for p_other in p_others:
            m.addConstr(gp.quicksum(x[i]*(ite*p_select[i] - p_other[i]) for i in coeff) >= gp.quicksum(org_coeff[i]*(p_other[i]- ite*p_select[i]) for i in coeff))
    #2 a1,a2,a3 하한과 상항
    m.addConstrs(org_coeff[i] + x[i] >= lbs[i] for i in coeff)
    m.addConstrs(org_coeff[i] + x[i] <= ubs[i] for i in coeff)
    if print_gurobi == False:
        m.setParam(GRB.Param.OutputFlag, 0)
    #풀이
    #m.params.NonConvex = 1
    m.Params.method = solver
    m.optimize()
    try:
        #print('Obj val: %g' % m.objVal)
        res = []
        for val in m.getVars():
            #print('VarName', val.VarName)
            if val.VarName[0] == 'x':
                res.append(float(val.x))
        return True, res
    except:
        #print('Infeasible')
        return False, None

def CustomersWithout(customers, except_ct):
    others = []
    others_name = []
    for customer in customers:
        if customer.name != except_ct.name:
            others.append(customer.u)
            others_name.append(customer.name)
    return others, others_name

def CustomersRemove(customers, name):
    for customer in customers:
        if name == customer.name:
            customers.remove(customer)
            break
    return customers


def CaseBSolver(rider, selected_ct, lbs, ubs, ite, past_data):
    selected =  selected_ct.u
    org_coeff = rider.p_coeff
    rev_ite = 1 + ite
    feasibility, thetas = CoeffRevise3(selected, org_coeff, lbs, ubs, rev_ite, past_data=past_data)
    rev_p_coeff = copy.deepcopy(rider.p_coeff)
    if feasibility == True:
        print("CaseB solver : Fea/ Theta:", thetas)
        index = 0
        for theta in thetas:
            rev_p_coeff[index] += theta
            index += 1
    else:
        print("CaseB solver : Inf")
    return rev_p_coeff


def InverseSolver(real_ct, exp_ct, rider, customer_t, ite, lbs, ubs, constraint_para = False):
    """
    문제를 풀어서 갱신된 rider의 p_coeff 값을 반환
    :param real_select: 실제 선택 고객 class
    :param p_select: 플랫폼이 선택한 고객 class
    :param rider: 주문을 선택한 라이더
    :param customer_t: 전체 고객 class list
    :param ite: 현재 누적된 데이터 수
    :param constraint_para:
    :return:
    """
    print("Diff")
    real_u = real_ct.u
    exp_u = exp_ct.u
    others, others_name = CustomersWithout(customer_t, real_ct)
    # 실행부 문제 풀기
    #feasibility, thetas = CoeffRevise(real_u, exp_u, others, rider.p_coeff, lbs, ubs, ite,
    #                                  constraint_para=constraint_para, print_gurobi=False)
    feasibility, thetas = CoeffRevise2(real_u, exp_u, others, rider.p_coeff, lbs, ubs, ite = 1, print_gurobi = False)
    rev_p_coeff = copy.deepcopy(rider.p_coeff)
    if feasibility == True:
        print('InverseSolver : Fea /Thetas : ', thetas)
        index = 0
        for theta in thetas:
            rev_p_coeff[index] += theta
            index += 1
    else:
        print("InverseSolver : Inf")
    return rev_p_coeff

def CaseSave(selecte_name, customers):
    others = []
    for ct in customers:
        if ct.name != customers[selecte_name].name:
            others.append(ct.u)
    data = [customers[selecte_name].u, others]
    return data


def printer(rider, customers):
    info1 = []
    info1.append(',')
    for value in rider.coeff:
        info1.append(value)
    for value in rider.p_coeff:
        info1.append(value)
    info1.append(',')
    info2 = []
    for customer in customers:
        tem = []
        tem.append(',')
        tem.append(customer.name)
        tem.append(',')
        for value in customer.u:
            tem.append(value)
        tem.append(',')
        info2.append(tem)
    print(info1)
    for i in info2:
        print(i)


def test(p_select, real_select):
    if p_select == None and real_select == None:
        print('A', p_select, real_select)
    elif p_select == None and real_select != None:
        print('B', p_select, real_select)
    elif p_select != None and real_select == None:
        print('C', p_select, real_select)
    elif p_select != real_select:
        print('D', p_select, real_select)
    else:
        print('E', p_select, real_select)