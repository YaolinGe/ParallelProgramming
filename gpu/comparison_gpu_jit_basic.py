import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time
from scipy.stats import norm

Ns = np.arange(10, 5000, 500)
MAX_ITER = 100


# == GPU
platforms = cl.get_platforms()
gpu_device = None
for platform in platforms:
    devices = platform.get_devices()
    for device in devices:
        if "AMD" in device.name:
            gpu_device = [device]
            break
print(gpu_device)

gpu_context = cl.Context(devices=gpu_device)
# gpu_context = cl.create_some_context()
gpu_queue = cl.CommandQueue(gpu_context)
gpu_mem_flags = cl.mem_flags

gpu_program = cl.Program(gpu_context, """
__kernel void gpu_cdf(__global const float *input, __global float *output)
{
    int i = get_global_id(0);
    output[i] = 0.5 * (1.0 + erf(input[i]));
}
""").build()


# == CPU
def cpu_cdf(x):
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = norm.cdf(x[i])
    return y


# == JIT
# @jit(nopython=True)
@jit
def jit_cdf(x):
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = norm.cdf(x[i])
    return y


# gpu_program = cl.Program(gpu_context, """
# __kernel void gpu_sqrt(__global const float *input, __global float *output)
# {
#     int i = get_global_id(0);
#     output[i] = sqrt(input[i]);
# }
# """).build()


# # == CPU
# def cpu_sqrt(x):
#     y = np.zeros_like(x)
#     for i in range(len(x)):
#         y[i] = np.sqrt(x[i])
#     return y
#
#
# # == JIT
# @jit(nopython=True)
# def jit_sqrt(x):
#     y = np.zeros_like(x)
#     for i in range(len(x)):
#         y[i] = np.sqrt(x[i])
#     return y


GPU_TIME = []
CPU_TIME = []
JIT_TIME = []


for N in Ns:
    x = np.linspace(-3, 3, N).astype(np.float32)
    # x = np.arange(N).astype(np.float32)
    gpu_input_buffer = cl.Buffer(gpu_context, gpu_mem_flags.READ_ONLY | gpu_mem_flags.COPY_HOST_PTR, hostbuf=x)
    gpu_output_buffer = cl.Buffer(gpu_context, gpu_mem_flags.WRITE_ONLY, x.nbytes)

    time_gpu = []
    time_cpu = []
    time_jit = []

    # == dummy setup
    # jit_cdf(x)
    gpu_result = np.empty_like(x)
    gpu_program.gpu_cdf(gpu_queue, x.shape, None, gpu_input_buffer, gpu_output_buffer)
    cl._enqueue_read_buffer(gpu_queue, gpu_output_buffer, gpu_result).wait()


    for i in range(MAX_ITER):


        # == GPU
        gpu_result = np.empty_like(x)
        t1 = time.time()
        gpu_program.gpu_cdf(gpu_queue, x.shape, None, gpu_input_buffer, gpu_output_buffer)
        cl._enqueue_read_buffer(gpu_queue, gpu_output_buffer, gpu_result).wait()
        t2 = time.time()
        time_gpu.append(t2 - t1)
        # == GPU


        # # == CPU
        # t1 = time.time()
        # cpu_result = cpu_cdf(x)
        # t2 = time.time()
        # time_cpu.append(t2 - t1)
        # # == CPU
        #
        #
        # # == JIT
        # t1 = time.time()
        # jit_result = jit_cdf(x)
        # t2 = time.time()
        # time_jit.append(t2 - t1)
        # # == JIT

    GPU_TIME.append([np.mean(time_gpu), np.std(time_gpu)])
    CPU_TIME.append([np.mean(time_cpu), np.std(time_cpu)])
    JIT_TIME.append([np.mean(time_jit), np.std(time_jit)])
    print("N=", N)
    print("GPU takes: ", np.mean(time_gpu))
    print("CPU takes: ", np.mean(time_cpu))
    print("JIT takes: ", np.mean(time_jit))
    print("GPU speed up: ", np.mean(time_cpu) / np.mean(time_gpu))
    print("JIT speed up: ", np.mean(time_cpu) / np.mean(time_jit))


# plt.figure()
# plt.plot(time_gpu, label='GPU')
# plt.plot(time_cpu, label='CPU')
# plt.plot(time_jit, label='JIT')
# plt.legend()
# plt.title('Function: sqrt comparison, vector size: '+str(N))
# plt.show()


# plt.figure()
# plt.subplot(131)
# plt.plot(gpu_result, label='GPU')
# plt.legend()
# plt.subplot(132)
# plt.plot(cpu_result, label='CPU')
# plt.legend()
# plt.subplot(133)
# plt.plot(jit_result, label='JIT')
# plt.legend()
# plt.show()


GPU_TIME = np.array(GPU_TIME)
CPU_TIME = np.array(CPU_TIME)
JIT_TIME = np.array(JIT_TIME)

plt.figure()
# plt.plot(Ns, GPU_TIME[:, 0], label="GPU")
# plt.title("GPU Time increase over vector size")
plt.errorbar(Ns, GPU_TIME[:, 0], yerr= GPU_TIME[:, 1], fmt="-o", capsize=5, label="GPU")
plt.errorbar(Ns, CPU_TIME[:, 0], yerr= CPU_TIME[:, 1], fmt="-o", capsize=5, label="CPU")
plt.errorbar(Ns, JIT_TIME[:, 0], yerr= JIT_TIME[:, 1], fmt="-o", capsize=5, label="JIT")
# plt.plot(Ns, GPU_TIME, label='GPU')
# plt.plot(Ns, CPU_TIME, label='CPU')
# plt.plot(Ns, JIT_TIME, label='JIT')
plt.xlabel("Vector size")
plt.legend()
plt.title('Time consumed for function: sqrt comparison')
plt.show()

#%%
import numpy as np
import pyopencl as cl

N = 5000000
MAX_ITER = 30
a_np = np.random.rand(N).astype(np.float32)
b_np = np.random.rand(N).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

prg = cl.Program(ctx, """
__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
""").build()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
knl = prg.sum  # Use this Kernel object for repeated calls
knl(queue, a_np.shape, None, a_g, b_g, res_g)

res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)


# CPU
def cpu_sum(a, b):
    c = np.empty_like(a)
    for i in range(len(a)):
        c[i] = a[i] + b[i]
    return c


# CPU
@jit(nopython=True)
def jit_sum(a, b):
    c = np.empty_like(a)
    for i in range(len(a)):
        c[i] = a[i] + b[i]
    return c


t_gpu = []
t_cpu = []
t_jit = []

jit_sum(a_np, b_np)


for i in range(MAX_ITER):
    # GPU
    t1 = time.time()
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    knl = prg.sum  # Use this Kernel object for repeated calls
    knl(queue, a_np.shape, None, a_g, b_g, res_g)
    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, res_g)
    t2 = time.time()
    t_gpu.append(t2 - t1)

    t1 = time.time()
    res_cpu = cpu_sum(a_np, b_np)
    t2 = time.time()
    t_cpu.append(t2 - t1)

    t1 = time.time()
    res_jit = jit_sum(a_np, b_np)
    t2 = time.time()
    t_jit.append(t2 - t1)

print("GPU:", np.mean(t_gpu))
print("CPU:", np.mean(t_cpu))
print("JIT:", np.mean(t_jit))
print("GPU speed up: ", np.mean(t_cpu) / np.mean(t_gpu))
print("JIT speed up: ", np.mean(t_cpu) / np.mean(t_jit))

print(np.sum(res_jit - res_cpu))
print(np.sum(res_cpu - res_np))
plt.plot(t_gpu, label='gpu')
plt.plot(t_cpu, label='cpu')
plt.plot(t_jit, label='jit')
plt.legend()
plt.show()
#%%
plt.plot(GPU_TIME[:, 0])
plt.show()


