# -*- coding: utf-8 -*-
import math
import random
from Basic_class import distance
import numpy as np


def InstanceGen(unit_dist= 8, std = 1, max_x = 50, max_y = 50, gen_n = 100):
    coordinates = []
    pool = np.random.normal(unit_dist, std, size= 10000)
    for counnt in range(gen_n):
        store_loc = [random.randrange(0,int(max_x/2)),random.randrange(0,int(max_y/2))]
        req_dist = random.choice(pool)
        angle = math.radian(random.randrange(0,360))
        customer_loc = [store_loc[0] + round(req_dist*math.sin(angle),4),store_loc[0] + round(req_dist*math.cos(angle),4) ]
        coordinates.append([store_loc, customer_loc])
    return coordinates

runtime = m.Runtime
print("The run time is %f" % runtime)
