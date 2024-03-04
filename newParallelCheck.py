from mpi4py import MPI
import pycuda.driver as cuda
import pycuda.autoinit  # This automatically initializes CUDA, but we still need to manage contexts manually
from pycuda.compiler import SourceModule
import os

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def gpu_work():
    # Simple kernel that does nothing but demonstrates usage
    kernel_code = """
    __global__ void dummy_kernel() {}
    """
    module = SourceModule(kernel_code)
    dummy_kernel = module.get_function("dummy_kernel")
    dummy_kernel(block=(1,1,1), grid=(1,1))

if __name__ == "__main__":
    cuda.init()  # Initialize CUDA driver

    hostname = os.uname()[1]
    print(f"Node {rank} of {size}, hostname: {hostname}, checking in with {cuda.Device.count()} GPUs.")

    for i in range(cuda.Device.count()):
        device = cuda.Device(i)  # Get reference to the i-th GPU device
        context = device.make_context()  # Create a context on the GPU
        # Additional properties
        name = device.name()
        compute_capability = device.compute_capability()
        total_memory = device.total_memory()
        attributes = device.get_attributes()

        # Example attribute: Number of multiprocessors
        multiprocessors = attributes[cuda.device_attribute.MULTIPROCESSOR_COUNT]

        print(f"Node {rank}, GPU {i}: {name}, Compute Capability: {compute_capability}, "
          f"Total Memory: {total_memory} bytes, Multiprocessors: {multiprocessors}")
        gpu_work()  # Perform a dummy operation on the GPU
        context.pop()

    comm.Barrier()  # Wait for all nodes to finish
    if rank == 0:
        print("All nodes have finished their GPU operations.")