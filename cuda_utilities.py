import tensorrt as trt
import numpy as np
import ctypes
from cuda import cudart

# Define the common module with CUDA utilities
class Common:
    @staticmethod
    def cuda_call(call):
        """
        Call CUDA function and check for errors.
        """
        result = call
        if result[0].value:
            raise RuntimeError(f"CUDA error: {result[0].value}")
        return result[1] if len(result) > 1 else None

    @staticmethod
    def memcpy_host_to_device(device_ptr, host_arr):
        """
        Copy data from host to device.
        """
        cudart.cudaMemcpy(
            device_ptr,  # device destination
            host_arr.ctypes.data,  # host source
            host_arr.nbytes,  # size
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,  # kind
        )

    @staticmethod
    def memcpy_device_to_host(host_arr, device_ptr):
        """
        Copy data from device to host.
        """
        cudart.cudaMemcpy(
            host_arr.ctypes.data,  # host destination
            device_ptr,  # device source
            host_arr.nbytes,  # size
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,  # kind
        )

# TEST Create common instance
#common = Common()