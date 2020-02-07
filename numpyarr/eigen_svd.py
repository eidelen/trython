
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import math

n = 1000
var_0 = 3.0
var_1 = 1.0
var_dir_0 = np.array([1, 1])
var_dir_1 = np.array([-1, 1])

data = np.zeros((2,n))
for i in range(0, n):
    v0 = random.gauss(0.0, math.sqrt(var_0))
    v1 = random.gauss(0.0, math.sqrt(var_1))
    offset = np.array([6,4])
    d0 = np.array(var_dir_0 / np.linalg.norm(var_dir_0)) * v0
    d1 = np.array(var_dir_1 / np.linalg.norm(var_dir_1)) * v1
    dc = (d0+d1)
    data[:,i] = dc + offset

covar_mat = np.cov(data)
print(covar_mat)

sample_mean = np.mean(data,axis=1)
data_norm = np.zeros((2,n))
for i in range(0, n):
    data_norm[:,i] = data[:,i] - sample_mean


cov = np.zeros((2,2))
for i in range(0,n):
    x = data_norm[:,i]
    c = np.outer(x,x)
    cov += c

cov_mat = 1.0/n * cov
print(cov_mat)

var, dire = np.linalg.eig(cov_mat)
print(var)
print(dire)



fig, ax = plt.subplots()
ax.scatter(data[0,:], data[1,:], marker='+', color='b')
ax.scatter(data_norm[0,:], data_norm[1,:], marker='+', color='g')
ax.axis('equal')
ax.grid()
plt.show()
