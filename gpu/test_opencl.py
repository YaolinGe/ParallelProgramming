
# Use OpenCL To Add Two Random Arrays (This Way Hides Details)
import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as pycl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
import numpy as np  # Import Numpy number tools
import matplotlib.pyplot as plt
from numba import jit
import time
from scipy.stats import norm
from scipy.spatial.distance import cdist



# == setup
N = 100
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
xx, yy = np.meshgrid(x, y)
xv = xx.reshape(-1, 1)
yv = yy.reshape(-1, 1)
grid = np.hstack((xv, yv))
distmatrix = cdist(grid, grid)
eta = 4.5 / .3
Sigma = 1 ** 2 * (1 + eta * distmatrix) * np.exp(-eta * distmatrix)
mu = np.abs(np.linalg.cholesky(Sigma) @ np.random.randn(Sigma.shape[0]).reshape(-1, 1)).astype(np.float32)
Variance = np.diag(Sigma).reshape(-1, 1).astype(np.float32)
threshold = np.ones_like(mu).astype(np.float32) * .5

# mu = np.zeros([N, 1]).astype(np.float32)
# Variance = np.ones_like(mu).astype(np.float32)
# threshold = np.linspace(-3, 3, N).astype(np.float32)
# ==

print("hello")
context = cl.create_some_context()  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue

mf = cl.mem_flags
mean = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mu)
vari = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Variance)
thres = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=threshold)
eibv = cl.Buffer(context, mf.WRITE_ONLY, mu.nbytes)

program = cl.Program(context, """
__kernel void eibv(__global const float *mean, __global const float *vari, __global const float *thres, __global float *eibv)
{
  int i = get_global_id(0);
  float temp = (thres[i] - mean[i]) / sqrt(2.0) / vari[i];
  float cdf = .5 * (1 + erf(temp));
  eibv[i] = cdf * (1 - cdf);
}
""").build()  # Create the OpenCL program

# program = cl.Program(context, """
# __kernel void eibv(__global const float *mean, __global const float *vari, __global const float *thres, __global float *eibv)
# {
#   int i = get_global_id(0);
#   float temp = (thres[i] - mean[i]) / sqrt(2.0) / vari[i];
#   float cdf = .5 * (1 + erf(temp));
#   eibv[i] = cdf;
# }
# """).build()  # Create the OpenCL program


Niter = 3
t_gpu = []
t_basic = []

print("=" * 60)
print("Start of GPU")

for i in range(Niter):
    eibv_np = np.empty_like(mu)
    t1 = time.time()
    program.eibv(queue, mu.shape, None, mean, vari, thres, eibv)  # Enqueue the program for execution and store the result in c
    cl._enqueue_read_buffer(queue, eibv, eibv_np).wait()
    EIBV = np.sum(eibv_np)
    t2 = time.time()
    t_gpu.append(t2 - t1)
print("GPU taks: ", np.mean(t_gpu))
print("End of GPU")
print("=" * 60)

def eibv_slow(mu, Sigma, threshold): # Function is compiled and runs in machine code
    # eibv = np.zeros_like(mu)
    eibv = 0
    for i in range(len(mu)):
        temp_cdf = norm.cdf(threshold[i], mu[i], Sigma[i])
        eibv += temp_cdf * (1 - temp_cdf)
    return eibv

# @jit
# # @register_callable(double(double))
# def eibv_fast(mu, Sigma, threshold): # Function is compiled and runs in machine code
#     eibv = np.zeros_like(mu)
#     for i in range(len(mu)):
#         eibv[i] = norm.cdf(threshold[i], mu[i], Sigma[i])
#     return eibv
# print("=" * 60)
# print("Start numba")
# eibv_fast(mu, Variance, threshold)
#
# t_numba = []
# for i in range(Niter):
#     t1 = time.time()
#     a3 = eibv_fast(mu, Variance, threshold)
#     t2 = time.time()
#     t_numba.append(t2 - t1)
# print("numba taks: ", np.mean(t_numba))
# print("End of numba")
# print("=" * 60)


print("=" * 60)
print("Start basic")
# a1 = np.arange(N)

for i in range(Niter):
    t1 = time.time()
    a2 = eibv_slow(mu, Variance, threshold)
    t2 = time.time()
    t_basic.append(t2 - t1)

print("basic taks: ", np.mean(t_basic))
print("End of basic")
print("=" * 60)


plt.plot(t_basic, label='basic')
plt.plot(t_gpu, label='gpu')
# plt.plot(t_numba, label='numba')
plt.legend()
plt.show()


print("Basic: ", np.mean(t_basic))
print("GPU Speed up: ", np.mean(t_basic) / np.mean(t_gpu))
# print("Numba Speed up: ", np.mean(t_basic) / np.mean(t_numba))

print("EIBV from GPU: ", EIBV)
print("EIBV from basic: ", a2)

# plt.plot(eibv_np, label='gpu')
# plt.plot(a2, label='basic')
# # plt.plot(a3, label='numba')
# plt.legend()
# plt.show()

# %%
# plt.subplot(121)
# plt.scatter(xv, yv, c=EIBV, s=150, cmap='BrBG')
# plt.colorbar()
# plt.subplot(122)
# plt.scatter(xv, yv, c=a2, s=150, cmap='BrBG')
# plt.colorbar()
# plt.show()
