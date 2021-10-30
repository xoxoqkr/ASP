# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt

info1  = 'test_ratio_file4'
# readline_test.py
f = open("test_ratio_file4.txt", 'r')
datas = []
indexs = list(range(0,1000))
lines = f.readlines()
A = [] #전체
for line in lines:
    #print(len(line))
    data = line.split(',')
    #print(len(data))
    tem = []
    count = 0
    tem.append(float(data[0][1:]))
    for number in data[1:len(data)-1]:
        #print(number)
        value = float(number)
        tem.append(value)
    end_index = data[-1].index(']')
    tem.append(float(data[-1][1:end_index]))
    A.append(tem)
    #print(len(tem))
f.close()

B = [] #error 가 원 값의 10% 이내
C = [] #error 가 원 값의 5% 이내
D = [] #error 가 원 값의 1% 이내
E = [] #error 가 원 값의 10% 보다 큰 경우

for data in A:
    #print(data[-1])
    if 0.9 <= data[-1] <= 1.1:
        B.append(data)
    if 0.95 <= data[-1] <= 1.05:
        C.append(data)
    if 0.99 <= data[-1] <= 1.01:
        D.append(data)
    if 0.9 >  data[-1] or data[-1] > 1.1:
        E.append(data)
print(len(A),len(B),len(C),len(D),len(E))
for i in A:
    plt.plot(indexs, i, linewidth = 0.5)
plt.xlabel("Round t")
plt.ylabel("Total error")
plt.savefig(info1 +'Fig 1 - a_dpi800.png',dpi = 800)
plt.clf() # Clear the current figure
for i in B:
    plt.plot(indexs, i, linewidth = 0.5)
plt.xlabel("Round t")
plt.ylabel("Total error")
plt.savefig(info1 +'Fig 1 - b_dpi800.png',dpi = 800)
plt.clf() # Clear the current figure
for i in C:
    plt.plot(indexs, i, linewidth = 0.5)
plt.xlabel("Round t")
plt.ylabel("Total error")
plt.savefig(info1 +'Fig 1 - c_dpi800.png',dpi = 800)
plt.clf() # Clear the current figure
for i in D:
    plt.plot(indexs, i, linewidth = 0.5)
plt.xlabel("Round t")
plt.ylabel("Total error")
plt.savefig(info1 +'Fig 1 - d_dpi800.png',dpi = 800)
plt.clf() # Clear the current figure
for i in E:
    plt.plot(indexs, i, linewidth = 0.5)
plt.xlabel("Round t")
plt.ylabel("Total error")
plt.savefig(info1 +'Fig 1 - e_dpi800.png',dpi = 800)
plt.clf() # Clear the current figure