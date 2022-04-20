# GPU programming thoughts

---

## Section I: Intro
### When not use GPU
- input is small
- calculation is too simple
- copy to/from GPU
- datatype is important, only use basic


### Tips
- cuda.jit(device=True), for scalar function, but cannot be called from GPU, not CPU host node.


---
## Section II: CUDA Kernel
- CUDA kernel(execution configuration)
- grid stride loops
- atomic operations avoid race condition


### Tips
- size of block, 32X threads, 128~512
- 20~40 blocks
- cuda-memcheck python ex.py
- cuda.jit(debug=True)

---
## Section III: Cuda submemory system
-


### Important!!!
- do not transfer data between CPU and GPU unless it is really necessary.
- do one dry run before launching the actual computation.
- synchronize before finishing.

# References
- [gufuncs](www.google.com)
- [Cuda best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [Cuda C](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction)
- [Cuda programming model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)
