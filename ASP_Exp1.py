# -*- coding: utf-8 -*-
from InstanceGen_class import InstanceGen
import random

re_new_type = False  # False: 전 날의 라이더 정보를 가져 옴.  True : 매일 매일 라이더 정보를 초기화 함.
input_para = True
type_nums = [8]
stds = [0.3]
LP_types = ['LP1']
betas = [1]
unit_dist = 8
max_x = 50
max_y = 50
incentive_time_ratio = 0.3 #0이면 LP3 모델 0<x<1이면 LP3_2모델
#cost_coeff = round(random.uniform(0.3,1.0),1)
#type_coeff = 3 - (1.5 + cost_coeff) #round(random.uniform(0.8,1.2),1)
#coeff = [cost_coeff,type_coeff,1.5] #[cost_coeff,type_coeff,1] #[1,1,1]


for beta in betas:
    for type_num in type_nums:
        for std in stds:
            std_input = std
            input_instances = InstanceGen(unit_dist=unit_dist, std=std_input, max_x=max_x, max_y=max_y, gen_n=10000)
            for ite in range(100):
                cost_coeff = round(random.uniform(0.3,1.0),1)
                type_coeff = 3 - (1.5 + cost_coeff) #round(random.uniform(0.8,1.2),1)
                coeff = [cost_coeff,type_coeff,1.5] #[cost_coeff,type_coeff,1] #[1,1,1]
                for LP_type in LP_types:
                    exec(open('ValueRevise_Run.py', encoding='UTF8').read(),
                         globals().update(type_num=type_num, std=std_input, LP_type=LP_type, beta=beta, input_para=input_para, input_instances = input_instances,
                                          rider_coeff = coeff, incentive_time_ratio = incentive_time_ratio))
                    #exec(open('Combine_Run.py', encoding='UTF8').read(),
                    #     globals().update(type_num=type_num, std=std_input, LP_type=LP_type, beta=beta,
                    #                      input_para=input_para, input_instances=input_instances, rider_coeff=coeff, re_new_type = re_new_type))
                    #input('실행 끝')
                    print(std_input, type_num, beta)
                #input('check')