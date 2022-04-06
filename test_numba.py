from numba import jit
import numpy as np
import time

x = np.arange(10000).reshape(100, 100)


@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting

t1 = time.time()
print(go_fast(x))
t2 = time.time()
print("time consumed: ", t2 - t1)

def go_slow(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting

t1 = time.time()
print(go_slow(x))
t2 = time.time()
print("time consumed: ", t2 - t1)


#%%
import numba as nb

@nb.njit
def sum_squares_2d_array_along_axis1(arr):
    res = np.empty(arr.shape[0], dtype=arr.dtype)
    for o_idx in range(arr.shape[0]):
        sum_ = 0
        for i_idx in range(arr.shape[1]):
            sum_ += arr[o_idx, i_idx] * arr[o_idx, i_idx]
        res[o_idx] = sum_
    return res


@nb.njit
def euclidean_distance_square_numba_v1(x1, x2):
    return -2 * np.dot(x1, x2.T) + np.expand_dims(sum_squares_2d_array_along_axis1(x1), axis=1) + sum_squares_2d_array_along_axis1(x2)

#%%
import numpy as np
import numba as nb
import time
import timeit


def just_sum(x2):
    return np.sum(x2, axis=1)

@nb.jit('double[:](double[:, :])', nopython=True)
def nb_just_sum(x2):
    return np.sum(x2, axis=1)

@nb.jit(nopython=True)
def nb_just_sum2(x2):
    return np.sum(x2, axis=1)

t_basic = []
t_numba1 = []
t_numba2 = []
Niter = 30
N = int(1e4)
for i in range(Niter):
    print(i)
    x2 = np.random.random((N, N))

    t1 = time.time()
    t = just_sum(x2)
    t2 = time.time()
    t_basic.append(t2 - t1)

    t1 = time.time()
    t = nb_just_sum(x2)
    t2 = time.time()
    t_numba1.append(t2 - t1)

    t1 = time.time()
    t = nb_just_sum2(x2)
    t2 = time.time()
    t_numba2.append(t2 - t1)


import matplotlib.pyplot as plt
plt.plot(t_basic)
plt.show()
plt.plot(t_numba1)
plt.show()
plt.plot(t_numba2)
plt.show()

print("basic: ", np.mean(t_basic))
print("numba1: ", np.mean(t_numba1))
print("numba2: ", np.mean(t_numba2))

# # code snippet to be executed only once
# mysetup = "import numba as nb"
#
# # code snippet whose execution time is to be measured
# mycode = '''
# def just_sum(x2):
#     return np.sum(x2, axis=1)
# '''
#
# print(timeit.timeit(setup = mysetup,
#                      stmt = mycode,
#                      number = 1000))
#
# # code snippet to be executed only once
# mysetup = '''
# import numba as nb
# import numpy as np
# '''
#
# mycode = '''
# @nb.jit('double[:](double[:, :])', nopython=True)
# def nb_just_sum(x2):
#     return np.sum(x2, axis=1)
# '''
#
# print(timeit.timeit(setup = mysetup,
#                      stmt = mycode,
#                      number = 10))
#
# mysetup = '''
# import numba as nb
# import numpy as np
# '''
#
# mycode = '''
# @nb.jit(nopython=True)
# def nb_just_sum2(x2):
#     return np.sum(x2, axis=1)
# '''
#
# print(timeit.timeit(setup = mysetup,
#                      stmt = mycode,
#                      number = 10))

#%%
from numba import jit
import numpy as np
import time
from scipy.stats import mvn
from scipy.spatial.distance import cdist

N = 50
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
xx, yy = np.meshgrid(x, y)
xv = xx.reshape(-1, 1)
yv = yy.reshape(-1, 1)
grid = np.hstack((xv, yv))
distmatrix = cdist(grid, grid)
eta = 4.5 / .6
Sigma = 1 ** 2 * (1 + eta * distmatrix) * np.exp(-eta * distmatrix)
plt.imshow(Sigma)
plt.colorbar()
plt.show()

mu = np.abs(np.linalg.cholesky(Sigma) @ np.random.randn(Sigma.shape[0]).reshape(-1, 1))

plt.scatter(xv, yv, c=mu, s=120, cmap="RdBu")
plt.colorbar()
plt.show()


#%%

Niter = 30

@jit
# @jit(nopython=True)
def eibv_numba(threshold, mu, Sigma): # Function is compiled and runs in machine code
    eibv = 0
    for i in range(len(mu)):
        eibv += mvn.mvnun(-np.inf, threshold, mu[i], Sigma[i, i])[0] - \
                mvn.mvnun(-np.inf, threshold, mu[i], Sigma[i, i])[0] ** 2
    return eibv


# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
eibv_numba(1, mu, Sigma)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))


# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
t_numba = []
for i in range(30):
    start = time.time()
    eibv_numba(1, mu, Sigma)
    end = time.time()
    t_numba.append(end - start)

print("numba = %s" % (np.mean(t_numba)))


def eibv_basic(threshold, mu, Sigma): # Function is compiled and runs in machine code
    eibv = np.zeros_like(mu)
    for i in range(len(eibv)):
        eibv[i] = mvn.mvnun(-np.inf, threshold, mu[i], Sigma[i, i])[0] - \
                  mvn.mvnun(-np.inf, threshold, mu[i], Sigma[i, i])[0] ** 2
    return eibv

    # return np.linalg.solve(a, np.ones([a.shape[0], 1]))
    # trace = 0.0
    # for i in range(a.shape[0]):
    #     trace += np.tanh(a[i, i])
    # return a + trace

t_basic = []
for i in range(Niter):
    start = time.time()
    eibv_basic(1, mu, Sigma)
    end = time.time()
    t_basic.append(end - start)

print("basic = %s" % (np.mean(t_basic)))

plt.plot(t_numba, label='numba')
plt.plot(t_basic, label='basic')
plt.legend()
plt.show()

