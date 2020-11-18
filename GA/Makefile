all: kernel.cu
	nvcc kernel.cu -o kernel.exe -lcudart -gencode arch=compute_35,code=sm_35 -gencode arch=compute_60,code=sm_60

# "-gencode arch=compute_35,code=sm_35" is required to THRUST library, e.g., device_vector work properly
# For code generation chaining documentation see https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#extended-notation
# For list of avaiable GPU architectures see https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list