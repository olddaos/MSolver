# test pycuda functionality with caffe_facade

import unittest

class TestPyCuda(unittest.TestCase):

    def test_axpy(self):
        """
        Test axpy function from scikits.cuda.cublas
        """
        import caffe_facade
        import scikits.cuda.cublas as cublas
        import numpy as np
        import pycuda.gpuarray as gpuarray
        from caffe_facade import pycuda_util

        caffe_facade.set_mode_gpu()
        caffe_facade.set_device(0)
        x = np.random.randn(5, 4, 3, 2).astype(np.float32)
        y = np.random.randn(5, 4, 3, 2).astype(np.float32)
        with pycuda_util.caffe_cuda_context():
            h = caffe_facade.cublas_handle()
            x_gpu = gpuarray.to_gpu(x)
            y_gpu = gpuarray.to_gpu(y)
            cublas.cublasSaxpy(h, x.size, 1.0, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
            y = x + y
            assert np.allclose(y_gpu.get(), y)

    def test_average(self):
        """
        Test average function on GPU
        """
        import caffe_facade
        import scikits.cuda.cublas as cublas
        import numpy as np
        import pycuda.gpuarray as gpuarray
        from caffe_facade import pycuda_util

        shape = (64, 32, 5, 5)
        num_elements = np.prod(shape)
        num_samples = 10

        data_cpu = np.zeros(shape, np.float32)
        data_cpu_received = [np.random.rand(*shape).astype(np.float32) for i in range(num_samples)]
        with pycuda_util.caffe_cuda_context():
            #GPU average
            data_gpu = gpuarray.to_gpu(np.zeros(shape, np.float32))
            h = caffe_facade.cublas_handle()
            data_gpu_temp = gpuarray.to_gpu(data_cpu_received[0])
            cublas.cublasScopy(h, num_elements, data_gpu_temp.gpudata, 1, data_gpu.gpudata, 1)
            for i in range(1, len(data_cpu_received)):
                data_gpu_temp = gpuarray.to_gpu(data_cpu_received[i])
                cublas.cublasSaxpy(h, num_elements, 1.0, data_gpu_temp.gpudata, 1, data_gpu.gpudata, 1)
            cublas.cublasSscal(h, num_elements, 1.0 / num_samples, data_gpu.gpudata, 1)

            #CPU average
            data_cpu = data_cpu_received[0] / num_samples
            for i in range(1, len(data_cpu_received)):
                data_cpu += data_cpu_received[i] / num_samples

            assert np.allclose(data_cpu, data_gpu.get())
