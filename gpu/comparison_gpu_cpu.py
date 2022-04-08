import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

Ns = np.arange(1, 2500, 50)
MAX_ITER = 100

GPU_TIME = []
CPU_TIME = []
JIT_TIME = []


for N in Ns:
    x = np.arange(N).astype(np.float32)

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

    gpu_input_buffer = cl.Buffer(gpu_context, gpu_mem_flags.READ_ONLY | gpu_mem_flags.COPY_HOST_PTR, hostbuf=x)
    gpu_output_buffer = cl.Buffer(gpu_context, gpu_mem_flags.WRITE_ONLY, x.nbytes)

    gpu_program = cl.Program(gpu_context, """
    __kernel void gpu_sqrt(__global const float *input, __global float *output)
    {
        int i = get_global_id(0);
        output[i] = sqrt(input[i]);
    }
    """).build()


    # == CPU
    def cpu_sqrt(x):
        y = np.zeros_like(x)
        for i in range(len(x)):
            y[i] = np.sqrt(x[i])
        return y


    # == JIT
    @jit(nopython=True)
    def jit_sqrt(x):
        y = np.zeros_like(x)
        for i in range(len(x)):
            y[i] = np.sqrt(x[i])
        return y


    time_gpu = []
    time_cpu = []
    time_jit = []

    # == dummy setup
    jit_sqrt(x)
    gpu_result = np.empty_like(x)
    gpu_program.gpu_sqrt(gpu_queue, x.shape, None, gpu_input_buffer, gpu_output_buffer)
    cl._enqueue_read_buffer(gpu_queue, gpu_output_buffer, gpu_result).wait()


    for i in range(MAX_ITER):


        # == GPU
        gpu_result = np.empty_like(x)
        t1 = time.time()
        gpu_program.gpu_sqrt(gpu_queue, x.shape, None, gpu_input_buffer, gpu_output_buffer)
        cl._enqueue_read_buffer(gpu_queue, gpu_output_buffer, gpu_result).wait()
        t2 = time.time()
        time_gpu.append(t2 - t1)
        # == GPU


        # == CPU
        t1 = time.time()
        cpu_result = cpu_sqrt(x)
        t2 = time.time()
        time_cpu.append(t2 - t1)
        # == CPU


        # == JIT
        t1 = time.time()
        jit_result = jit_sqrt(x)
        t2 = time.time()
        time_jit.append(t2 - t1)
        # == JIT

    GPU_TIME.append([np.mean(time_gpu), np.std(time_gpu)])
    CPU_TIME.append([np.mean(time_cpu), np.std(time_cpu)])
    JIT_TIME.append([np.mean(time_jit), np.std(time_jit)])
    print("N=", N)
    # print("GPU takes: ", np.mean(time_gpu))
    # print("CPU takes: ", np.mean(time_cpu))
    # print("JIT takes: ", np.mean(time_jit))
    # print("GPU speed up: ", np.mean(time_cpu) / np.mean(time_gpu))
    # print("JIT speed up: ", np.mean(time_cpu) / np.mean(time_jit))


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


