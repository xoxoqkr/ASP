# -*- coding: utf-8 -*-

import Revise_test_CL as Basic
import copy
import Revise_test_Runner as Runner


## 파라메터 부###
num_rider = 1
num_order_num = 10
attribute_num = 3
mu = 2
sigma = 0.5
mus = [1.2, -1, -300]
sigmas = [0.2, 0.2, 100]
lbs = []
ubs = []
for index in range(len(mus)):
    lbs.append(mus[index] - 2*sigmas[index])
    ubs.append(mus[index] + 2*sigmas[index])
pool_size = 1000
dist_type = 'normal'
Run_T = 1000
constraint_para = False #제약식
minimum_value = 0  # 이 값보다 적은 주문을 라이더는 선택하지 X
maximum_value = 100
ITE_NUM = 1
for _ in range(ITE_NUM):
    ###rider 생성부 ###
    Org_theta_list = []
    Rider_LIST = []
    for i in range(num_rider):
        #coeff = Basic.RandomData(mu, sigma, 1000, 1, num_coeff = attribute_num ,type = 'normal')
        coeff = []
        for attribute_index in range(attribute_num):
            value = Basic.RandomData2(mus, sigmas, 1000, type='normal', positive= False)
            coeff.append(value)
        p_coeff = [mus]#Basic.RandomData(mu, sigma, 1000, 1, num_coeff = attribute_num ,type = 'normal')#[[1,1]] #RandomData(0, 1, 1000, 1, num_coeff = attribute_num,type = 'normal')
        print('Real theta', coeff[0], '/Exp theta', p_coeff[0])
        rider = Basic.Rider(i, coeff[0], p_coeff[0])
        Rider_LIST.append(rider)
        Org_theta_list.append([i, coeff[0],copy.deepcopy(p_coeff[0])])
    rider = Rider_LIST[0]
    saved_data = [mu,sigma]
    saved_ratio = [mu,sigma]
    for i in rider.coeff + rider.p_coeff:
        saved_data.append(round(i, 4))
    tem1 = []
    count = 0
    for i in range(len(rider.p_coeff)):
        tem1.append(rider.coeff[i]/rider.p_coeff[i])
    saved_ratio.append(round(sum(tem1)/3.0,4))
    ##실험부###
    #saved_data, saved_ratio = Runner.SystemRunner(Rider_LIST, Run_T, num_order_num, mu, sigma, pool_size, minimum_value, maximum_value, dist_type, constraint_para)
    #saved_data, saved_ratio = Runner.SystemRunner2(Rider_LIST, Run_T, num_order_num, mu, sigma, pool_size, minimum_value, lbs, ubs, dist_type, constraint_para)
    saved_data, saved_ratio = Runner.SystemRunner3(Rider_LIST, Run_T, num_order_num, minimum_value, lbs, ubs, constraint_para)

    index = 0
    for info in Org_theta_list:
        rider = Rider_LIST[index]
        print('Org info',Org_theta_list[index])
        print('Rider:',rider.name,'/theta :', rider.coeff,'/->theta', rider.p_coeff)
        res2 = []
        for count in range(len(Org_theta_list[index][2])):
            res2.append(round(Org_theta_list[index][2][count]/rider.coeff[count],4))
        print('start theta times ::', res2)
        res2 = []
        for count in range(len(rider.coeff)):
            res2.append(round(rider.p_coeff[count]/rider.coeff[count],4))
        print('end theta times ::', res2)
        index += 1
    ##결과 출력부##
    # writedata.py
    f = open("ThetaRevise/test_file4.txt", 'a')
    f.write(str(saved_data) + "\n")
    f.close()
    f = open("ThetaRevise/test_ratio_file4.txt", 'a')
    f.write(str(saved_ratio) + "\n")
    f.close()


