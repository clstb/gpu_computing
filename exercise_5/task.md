## Heat Flow Simulation

In this exercise, you will implement a CUDA kernel that performs a 2D heat flow simulation using a iterative stencil kernel.
For this you start with a sequential C++ version of the code, that we have provided.

Your task is to re-write the simulation into a CUDA implementation that takes advantage of the simulations parallelism.
For the purpose of this exercise you may assume a square shaped matrix as input.
We recommend utilizing a double buffering approach with a `current` and `next` buffer that get swapped after each iteration of the simulation, as shown in the CPU version.
For full credits you should take advantage of the shared memory in the kernel for improved memory access and not perform unnecessary data transfers between CPU and GPU.

### Evaluation Criteria:

Correctness: The output should be a working CUDA accelerated 2D heat flow simulation that works with various inputs sizes, flow coefficients and cell sizes.
Code quality: Clear, well-structured, and documented code.
