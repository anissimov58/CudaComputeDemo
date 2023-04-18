# CudaComputeDemo
A small demo of how to perform CUDA compute and why this is not always the optimal choice on small amounts of data.

TL, DR: If you can split your work in less than number of cores (threads) on your CPU - do it on your CPU. If your work can be split in more than 1.25*cores(threads) (it can be highly parallelized) on our CPU - it’s worth it do on CUDA.  

Feel free to use this code as an example or a starting point in CUDA compute.

WARNING! This code is not optimized. This code only meant to show BASIC acceleration using CUDA compared to CPU. By doing further optimizations, both CUDA and CPU can be optimized. CPU code is purposefully NOT multithreaded. By implementing multithreading its compute time can be decreased by the factor of time/cores * 90% or if you CPU supports multithreading (most of them do) by about time/threads * 70% depending on code implementation.

Sample Data: calculation of 16k and 64k samples of Y[i] = A[i]*B[i] + A[i]* A[i]* A[i]*C[i]/B[i] on CPU and on CUDA.

CPU: Intel Core i5-8500, 3.90 GHz (6 cores, 6 threads)
RAM: 16 Gb DDR4 2666 MHz
GPU: nVidia GeForce GTX 970
•	CUDA version: 5.2
•	CUDA cores: 1664
•	VRAM: 4095 MB
•	Max Block Size: 1024*1024*64
•	Max Warp Size: 32
•	Max Grid Size: 2147483647*65535*65535
OS Win10 Pro 1903 


Vertical axis - time to compute in ms.
Horizontal axis - block size on CUDA. 

Acceleration on 16k samples: ![image](https://user-images.githubusercontent.com/102085875/232715266-ef0abd1a-6ea8-4ee6-91ad-1741daa61335.png)
Acceleration on 64k samples: ![image](https://user-images.githubusercontent.com/102085875/232715338-b22d7194-7d82-4c3c-9643-60b279edde3e.png)
