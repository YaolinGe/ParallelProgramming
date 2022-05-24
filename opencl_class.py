import numpy as np
import pyopencl as cl

N = 50000
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
knl = prg.sum
knl(queue, a_np.shape, None, a_g, b_g, res_g)
res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)
print(res_np - (a_np + b_np))
print(np.linalg.norm(res_np - (a_np + b_np)))
assert np.allclose(res_np, a_np + b_np)




#%%
#Port from Adventures in OpenCL Part1 to PyOpenCL
# http://enja.org/2010/07/13/adventures-in-opencl-part-1-getting-started/
# http://documen.tician.de/pyopencl/

import pyopencl as cl
import numpy

class CL:
    def __init__(self):
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

    def loadProgram(self, filename):
        #read in the OpenCL source file as a string
        f = open(filename, 'r')
        fstr = "".join(f.readlines())
        print(fstr)
        #create the program
        self.program = cl.Program(self.ctx, fstr).build()

    def popCorn(self):
        mf = cl.mem_flags

        #initialize client side (CPU) arrays
        self.a = numpy.array(range(10), dtype=numpy.float32)
        self.b = numpy.array(range(10), dtype=numpy.float32)

        #create OpenCL buffers
        self.a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.a)
        self.b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.b)
        self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.b.nbytes)

    def execute(self):
        self.program.part1(self.queue, self.a.shape, None, self.a_buf, self.b_buf, self.dest_buf)
        c = numpy.empty_like(self.a)
        # cl.enqueue_read_buffer(self.queue, self.dest_buf, c).wait()
        print("a", self.a)
        print("b", self.b)
        print("c", c)



if __name__ == "__main__":
    example = CL()
    example.loadProgram("ParallelProgramming/part1.cl")
    example.popCorn()
    example.execute()

