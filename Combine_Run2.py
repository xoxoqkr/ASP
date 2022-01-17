# -*- coding: utf-8 -*-


datas = ['데이터/new_data_1','데이터/new_data_2_RandomCluster','데이터/new_data_2_Random']
for data_dir in datas:
    exec(open('Basic_run.py', encoding='UTF8').read(),globals().update(data_dir=data_dir))