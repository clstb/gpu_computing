# Convolution via WMMA (Tensor Cores)

In this exercise, you will implement a **1D convolution** using **Tensor Core WMMA instructions**.

For are given list of kernels (a fixed-size array of weights), your task is to convolve each kernel with a large input array.

### Idea

To compute a single convolution, we perform a dot-product between the kernel and an equally sized window, that we slide across the input:
```
for (i = 0; 0 < output_size; i++)
    acc = 0
    for (k = 0; 0 < kernel_size; k++)
        acc += kernel[k] * input[i + k]
    output[i] = acc
```

When computing multiple convolutions with multiple kernels at once, we can perform the convolutions together as a matrix multiplication. For that, we break the input into **chunks of consecutive windows**:

* For each chunk, stack all windows into a matrix *A* of size *CHUNK_SIZE × KERNEL_SIZE*.
* Stack all kernels into a matrix *B* of size *KERNEL_SIZE × NUM_KERNELS*.
* Multiplying *A × B* gives the convolution results for all windows of this chunk.

### Assumptions for simplification

* You can use blocks of size 32, so that each block only consists of a single warp.
* The sizes in the template are chosen so that they fit perfectly into a sequence of 16x16x16 matrix multiplications.

When renting a GPU from vast.ai, remember to check that they have tensor cores.
You can use https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units to check.
For each GPU the Wikipedia article shows the "Core Configuration", and a footnote explains if this contains Tensor Cores or not.

The provided `.cu` file gives you the baseline structure to complete.
