import pyopencl as cl
import numpy as np
import time
import matplotlib.pyplot as plt


class GPUParallelComputing:

    def __init__(self):
        self.gpu_context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.gpu_context)

    def load_program_kernel(self, filename):
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        self.program = cl.Program(self.gpu_context, fstr).build()


    def set_up_parallel_computing(self):
        mf = cl.mem_flags
        N = 100000000
        #initialize client side (CPU) arrays
        self.input_data1 = np.random.rand(N).astype(np.float32)
        self.input_data2 = np.random.rand(N).astype(np.float32)
        # self.a = numpy.array(range(10), dtype=numpy.float32)
        # self.b = numpy.array(range(10), dtype=numpy.float32)

        #create OpenCL buffers
        self.input_data1_buffer = cl.Buffer(self.gpu_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.input_data1)
        self.input_data2_buffer = cl.Buffer(self.gpu_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.input_data2)
        # self.a_buf = cl.Buffer(self.gpu_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.a)
        # self.b_buf = cl.Buffer(self.gpu_context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.b)
        self.output_data_buffer = cl.Buffer(self.gpu_context, mf.WRITE_ONLY, self.input_data1.nbytes)
        # self.dest_buf = cl.Buffer(self.gpu_context, mf.WRITE_ONLY, self.b.nbytes)

    def execute(self):
        Niter = 30
        t_gpu = []
        t_basic = []
        t_numba = []
        for i in range(Niter):
            print(i)

            t1 = time.time()
            self.program.inverse(self.queue, self.input_data1.shape, None, self.input_data1_buffer,
                                 self.input_data2_buffer, self.output_data_buffer)
            self.output_data = np.empty_like(self.input_data1)
            cl.enqueue_copy(self.queue, self.output_data, self.output_data_buffer)
            t2 = time.time()
            t_gpu.append(t2 - t1)

            t1 = time.time()
            c = self.input_data1 * self.input_data2
            t2 = time.time()
            t_basic.append(t2 - t1)

        plt.plot(t_basic, label='basic')
        plt.plot(t_gpu, label='gpu')
        plt.legend()
        plt.show()

        print("Basic: ", np.mean(t_basic))
        print("GPU: ", np.mean(t_gpu))

        # print("Input: ", self.input_data)
        # print("Output: ", self.output_data)


if __name__ == "__main__":
    gpu = GPUParallelComputing()
    gpu.load_program_kernel("ParallelProgramming/eibv/eibv_kernel.cl")

    gpu.set_up_parallel_computing()
    gpu.execute()

