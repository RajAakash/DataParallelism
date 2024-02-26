from mpi4py import MPI
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def gpu_work():
    # Simple kernel that does nothing but demonstrates usage
    kernel_code = """
    __global__ void dummy_kernel()
    {
    }
    """
    module = SourceModule(kernel_code)
    dummy_kernel = module.get_function("dummy_kernel")
    dummy_kernel(block=(1,1,1), grid=(1,1))

if __name__ == "__main__":
    cuda.init() # Initialize CUDA driver

    print(f"Node {rank} of {size} checking in with {cuda.Device.count()} GPUs.")

    for i in range(cuda.Device.count()):
        cuda.Device(i).make_context() # Create a context on the GPU
        print(f"Node {rank}, GPU {i}: Performing dummy operation.")
        gpu_work() # Perform a dummy operation on the GPU
        cuda.Context.pop() # Remove the current context

    comm.Barrier() # Wait for all nodes to finish
    if rank == 0:
        print("All nodes have finished their GPU operations.")

