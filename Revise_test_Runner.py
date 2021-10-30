# -*- coding: utf-8 -*-

import math
import Revise_test_CL as RCL
import gurobipy as gp
from gurobipy import GRB
import copy


def BetaSolver(coeff, d_m,l,u, print_gurobi = False, solver = 1):
    num_coeff = len(coeff)
    coeff_range = list(range(num_coeff))
    # D.V. and model set.
    m = gp.Model("betasolver")
    print(l, coeff[0], u,coeff[0])
    lower = round(l/coeff[0],4)
    upper = round(u/coeff[0],4)
    beta = m.addVar(lb = lower, ub = upper, vtype=GRB.CONTINUOUS, name="beta")
    #목적식
    m.setObjective(beta*beta, GRB.MAXIMIZE)
    #m.setObjective(beta , GRB.MINIMIZE)
    for case in d_m:
        selected = case[0]
        others = case[1]
        for other in others:
            m.addConstr(gp.quicksum(coeff[i] * selected[i] for i in coeff_range) - beta*coeff[0]*selected[0] >=
                        gp.quicksum(coeff[i] * other[i] for i in coeff_range) - beta*coeff[0]*other[0])
    if print_gurobi == False:
        m.setParam(GRB.Param.OutputFlag, 0)
    #풀이
    #m.params.NonConvex = 1
    m.Params.method = solver
    m.optimize()
    try:
        for val in m.getVars():
            if val.VarName[0] == 'b':
                res = float(val.x)
        return True, res
    except:
        #print('Infeasible')
        return False, 1

def SystemRunner2(riders, Run_T,num_order_num, mu, sigma, pool_size, min_value, lbs, ubs , dist_type, constraint_para):
    inf_infos = []
    saved_data2 = []
    saved_ratio = []
    saved_data = {'A': [], 'B': [], 'C': [], 'D': [], 'E': []}
    for t in range(1, Run_T + 1):
        #customer_t = CustomerGen(num_order_num, mu, sigma, pool_size, num_coeff = len(riders[0].coeff),type = dist_type, t = t)
        customer_t = RCL.CustomerGen2(num_order_num, t)
        ##플랫폼의 예상##
        ite = round(1.0 / math.sqrt(t), 4)
        for rider in riders:
            p_exp, p_select, p_values, p_rank = RCL.SingleRun(rider, customer_t, type='platform', min_value = min_value)
            real, real_select, real_values, r_rank = RCL.SingleRun(rider, customer_t, type='rider', min_value = min_value)
            #diff = Comapre2List(p_exp, real)
            if p_select == None and real_select == None: #A
                pass #알수 있는 정보가 없음.
            elif p_select == None and real_select != None: #B
                #value 계수 변화 필요
                print("ITE", t, "TEST", p_values, real_values, "Select", p_select, real_select, )
                print("플랫폼 랭크", p_rank)
                print("라이더 랭크", r_rank)
                rider.p_coeff = RCL.CaseBSolver(rider, customer_t[real_select], lbs, ubs, ite, saved_data['E'])
            elif p_select != None and real_select == None: #C
                pass #관측할 수 없음
            elif p_select != real_select: #D
                #Inverse 문제 풀기
                past_data1 = saved_data['D'][int(len(saved_data['D'])*0.9):] #최신의 자료 만을 사용하려는 방식
                past_data2 = saved_data['E'][int(len(saved_data['E'])*0.9):]
                print("다른 결과")
                print("ITE", t, "TEST", p_values, real_values, "Select", p_select, real_select, )
                print("라이더 가치", rider.coeff, "플랫폼 가치", rider.p_coeff)
                print("플랫폼 랭크", p_rank, len(past_data1))
                print("라이더 랭크", r_rank, len(past_data2))
                RCL.printer(rider, customer_t)
                print(rider.coeff,",",rider.p_coeff)
                for customer in customer_t:
                    print(customer.name, ",", customer.u)
                rider.p_coeff = RCL.InverseSolver(customer_t[real_select], customer_t[p_select], rider, customer_t, ite, lbs, ubs, constraint_para=constraint_para)
                #rider.p_coeff = CaseBSolver(rider, customer_t[real_select], lbs, ubs, ite, past_data1 + past_data2)
                data = RCL.CaseSave(real_select, customer_t)
                saved_data['D'].append(data)
            else: #E -> 정보 저장
                print("일치하는 결과")
                print("ITE", t, "TEST", p_values, real_values, "Select", p_select, real_select, )
                RCL.printer(rider, customer_t)
                print(rider.coeff,",",rider.p_coeff)
                for customer in customer_t:
                    print(customer.name, ",", customer.u)
                data = RCL.CaseSave(real_select, customer_t)
                saved_data['E'].append(data)
                pass #알 수 있는 정보가 없음
            ##선택된 고객 제거
            if real_select != None:
                customer_t = RCL.CustomersRemove(customer_t, real_select)
            for customer in customer_t:
                if customer.gen_t < t - 10 :
                    customer_t.remove(customer)
            if t % 100 == 0:
                tem_data = []
                for i in rider.p_coeff:
                    tem_data.append(round(i,4))
                saved_data2 += tem_data
            tem1 = []
            for i in range(len(rider.p_coeff)):
                tem1.append(rider.p_coeff[i] / rider.coeff[i])
            saved_ratio.append(round(sum(tem1) / 3.0, 4))
    return saved_data2, saved_ratio

def SystemRunner3(riders, Run_T,num_order_num, min_value, lbs, ubs , constraint_para):
    saved_data2 = []
    saved_ratio = []
    saved_data = {'A': [], 'B': [], 'C': [], 'D': [], 'E': []}
    for t in range(1, Run_T + 1):
        #customer_t = CustomerGen(num_order_num, mu, sigma, pool_size, num_coeff = len(riders[0].coeff),type = dist_type, t = t)
        customer_t = RCL.CustomerGen2(num_order_num, t)
        ##플랫폼의 예상##
        ite = round(1.0 / math.sqrt(t), 4)
        beta_update = False
        divider = 1
        for rider in riders:
            p_exp, p_select, p_values, p_rank = RCL.SingleRun(rider, customer_t, type='platform', min_value = min_value)
            real, real_select, real_values, r_rank = RCL.SingleRun(rider, customer_t, type='rider', min_value = min_value)
            if p_select == None and real_select == None: #A
                pass #알수 있는 정보가 없음.
            elif p_select == None and real_select != None: #B
                #value 계수 변화 필요
                print("ITE", t, "TEST", p_values, real_values, "Select", p_select, real_select, )
                print("플랫폼 랭크", p_rank)
                print("라이더 랭크", r_rank)
                rider.p_coeff = RCL.CaseBSolver(rider, customer_t[real_select], lbs, ubs, ite, saved_data['E'])
            elif p_select != None and real_select == None: #C
                pass #관측할 수 없음
            elif p_select != real_select: #D
                #Inverse 문제 풀기
                past_data1 = saved_data['D'][int(len(saved_data['D'])*0.9):] #최신의 자료 만을 사용하려는 방식
                past_data2 = saved_data['E'][int(len(saved_data['E'])*0.9):]
                print("다른 결과")
                print("ITE", t, "TEST", p_values, real_values, "Select", p_select, real_select, )
                print("라이더 가치", rider.coeff, "플랫폼 가치", rider.p_coeff)
                print("플랫폼 랭크", p_rank, len(past_data1))
                print("라이더 랭크", r_rank, len(past_data2))
                #RCL.printer(rider, customer_t)
                print(rider.coeff,",",rider.p_coeff)
                #for customer in customer_t:
                #    print(customer.name, ",", customer.u)
                rider.p_coeff = RCL.InverseSolver(customer_t[real_select], customer_t[p_select], rider, customer_t, ite, lbs, ubs, constraint_para=constraint_para)
                #rider.p_coeff = CaseBSolver(rider, customer_t[real_select], lbs, ubs, ite, past_data1 + past_data2)
                data = RCL.CaseSave(real_select, customer_t)
                saved_data['D'].append(data)
            else: #E -> 정보 저장
                data = RCL.CaseSave(real_select, customer_t)
                saved_data['E'].append(data)
                print("일치하는 결과")
                print("ITE", t, "TEST", p_values, real_values, "Select", p_select, real_select)
                #RCL.printer(rider, customer_t)
                #print(rider.coeff,",",rider.p_coeff)
                #for customer in customer_t:
                #    print(customer.name, ",", customer.u)
                """
                print('정보1',rider.p_coeff, lbs[0], ubs[0])
                input('확인')
                beta_fea, beta = BetaSolver(rider.p_coeff, saved_data['E'], lbs[0], ubs[0])                
                if beta_update == True:
                    # 라이더의 수수료를 낮춰서 제안함
                    org_value = 0
                    for index in range(len(rider.old_coeff)):
                        org_value += customer_t[p_select]*rider.old_coeff[index]
                    gamma = org_value/p_values
                    print('감마는', gamma)
                    for index in range(len(rider.p_coeff)):
                        rider.p_coeff[index] = gamma*rider.p_coeff[index]
                    beta_update = False
                print("베타 결과",beta_fea, beta)
                if beta_fea == True and beta != 1:
                    beta_update = True
                    print("베타는", beta)
                    rider.old_coeff = copy.deepcopy(rider.p_coeff)
                    rider.p_coeff[0] = beta*rider.p_coeff[0]                
                """
                pass #알 수 있는 정보가 없음
            ##선택된 고객 제거
            if real_select != None:
                customer_t = RCL.CustomersRemove(customer_t, real_select)
            for customer in customer_t:
                if customer.gen_t < t - 10 :
                    customer_t.remove(customer)
            if t % 100 == 0:
                tem_data = []
                for i in rider.p_coeff:
                    tem_data.append(round(i,4))
                saved_data2 += tem_data
            tem1 = []
            for i in range(len(rider.p_coeff)):
                tem1.append(rider.p_coeff[i] / rider.coeff[i])
            saved_ratio.append(round(sum(tem1) / 3.0, 4))
    return saved_data2, saved_ratio