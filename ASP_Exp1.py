# -*- coding: utf-8 -*-
from InstanceGen_class import InstanceGen
import random

input_para = True
type_nums = [2,5,8,11,14,17]
stds = [1,2,3]
LP_types = ['LP2','LP1']
betas = [1,0.5,0.4,0.3]
unit_dist = 8
max_x = 50
max_y = 50
cost_coeff = round(random.uniform(0.3,1.0),1)
type_coeff = 3 - (1.5 + cost_coeff) #round(random.uniform(0.8,1.2),1)
coeff = [cost_coeff,type_coeff,1.5] #[cost_coeff,type_coeff,1] #[1,1,1]



for std in stds:
    for type_num in type_nums:
        input_instances = InstanceGen(unit_dist=unit_dist, std=std, max_x=max_x, max_y=max_y, gen_n=10000)
        for beta in betas:
            for LP_type in LP_types:
                exec(open('ValueRevise_Run.py', encoding='UTF8').read(),
                     globals().update(type_num=type_num, std=std, LP_type=LP_type, beta=beta, input_para=input_para, input_instances = input_instances, rider_coeff = coeff))
            input('실행 끝')
"""
type_num = 5
std = 2
LP_type = 'LP2'
beta = 0.3
input_para = True
exec(open('ValueRevise_Run.py', encoding='UTF8').read(),globals().update(type_num = type_num, std = std,LP_type = LP_type, beta = beta, input_para = input_para))
"""
