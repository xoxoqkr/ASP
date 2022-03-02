# -*- coding: utf-8 -*-
from InstanceGen_class import InstanceGen
import random

re_new_type = False  # False: 전 날의 라이더 정보를 가져 옴.  True : 매일 매일 라이더 정보를 초기화 함.
input_para = True
selected_nonnegative = True #True: 선택한 주문의 가치를 0보다 크게 / False: 선택한 주문의 가치 크기가 음수가 되어도 가능
type_nums = [8]
stds = [0.3]
LP_types = ['LP1','LP2']
betas = [1]
unit_dist = 8
max_x = 50
max_y = 50
incentive_time_ratio = 0.9 #0이면 LP3 모델 0<x<1이면 LP3_2모델->보조금으로 해공간을 잘라보는 모형
#cost_coeff = round(random.uniform(0.3,1.0),1)
#type_coeff = 3 - (1.5 + cost_coeff) #round(random.uniform(0.8,1.2),1)
#coeff = [cost_coeff,type_coeff,1.5] #[cost_coeff,type_coeff,1] #[1,1,1]


for beta in betas:
    for type_num in type_nums:
        for std in stds:
            std_input = std
            input_instances = InstanceGen(unit_dist=unit_dist, std=std_input, max_x=max_x, max_y=max_y, gen_n=10000)
            input_instances = None
            performance = [[],[]]
            for LP_type in LP_types:
                f3 = open("LP3 보조금 유무 정리.txt", 'a')
                f3.write(LP_type + "시작 \n")
                f3.close()
                for ite in range(200):
                    cost_coeff = round(random.uniform(0.6,1.1),1)
                    type_coeff = round(random.uniform(0.5,0.9),1)
                    fee_coeff = 3 - (type_coeff + cost_coeff)
                    coeff = [cost_coeff,type_coeff,fee_coeff] #[cost_coeff,type_coeff,1] #[1,1,1]
                    for subsiduyforLP3 in [True, False]:
                        exec(open('ValueRevise_Run.py', encoding='UTF8').read(),
                             globals().update(type_num=type_num, std=std_input, LP_type=LP_type, beta=beta, input_para=input_para, input_instances = input_instances,
                                              rider_coeff = coeff, incentive_time_ratio = incentive_time_ratio,subsiduyforLP3 = subsiduyforLP3, selected_nonnegative = selected_nonnegative, performance_measure = performance))
                        #print(std_input, type_num, beta)
                        #input('check')
                        #print(performance)
                        #input('확인')
                tem_print = [[],[]]
                save_data = [[],[]]
                f3 = open("LP3 보조금 유무 정리.txt", 'a')
                for index in range(len(performance[0])):
                    last_index = min(len(performance[0][index]),len(performance[1][index])) - 1
                    tem_print[0].append(performance[0][index][last_index])
                    tem_print[1].append(performance[1][index][last_index])
                    save_data[0].append(performance[0][index])
                    save_data[1].append(performance[1][index])
                saved_head = ['T','F']
                head_count = 0
                for infos in save_data:
                    count = 0
                    head_name = saved_head[head_count]
                    for info in infos:
                        f3.write(head_name + str(info) + '\n')
                        count += 1
                    head_count += 1
                #f3.write('T' + str(tem_print[0]) + '\n')
                #f3.write('F' + str(tem_print[1]) + '\n')
                f3.close()
                print(tem_print)
                print('보조금 지급 평균 : {}/ 보조금 지급X평균:{}'.format(round(sum(tem_print[0])/len(tem_print[0]),4),round(sum(tem_print[1])/len(tem_print[1]),4)))