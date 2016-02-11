import unittest


def simple_net_file(num_output):
    """
    Create a simple net prototxt, based on test_net.cpp, returning the name
    of the (temporary) file.
    """
    import tempfile

    f = tempfile.NamedTemporaryFile(delete=False)
    f.write("""
            name: 'testnet'
            force_backward: true
            layer {
                type: 'DummyData'
                name: 'data'
                top: 'data'
                top: 'label'
                dummy_data_param {
                    num: 5
                    channels: 2
                    height: 3
                    width: 4
                    num: 5
                    channels: 1
                    height: 1
                    width: 1
                    data_filler {
                        type: 'gaussian'
                        std: 1
                    }
                    data_filler {
                        type: 'constant'
                    }
                }
            }
            layer {
                type: 'Convolution'
                name: 'conv'
                bottom: 'data'
                top: 'conv'
                convolution_param {
                    num_output: 11
                    kernel_size: 2
                    pad: 3
                    weight_filler {
                        type: 'gaussian'
                        std: 1
                    }
                    bias_filler {
                        type: 'constant'
                        value: 2
                    }
                }
                param {
                    decay_mult: 1
                }
                param {
                    decay_mult: 0
                }
            }
            layer {
                type: 'InnerProduct'
                name: 'ip'
                bottom: 'conv'
                top: 'ip'
                inner_product_param {
                    num_output: """ + str(num_output) + """
                    weight_filler {
                        type: 'gaussian'
                        std: 2.5
                    }
                    bias_filler {
                        type: 'constant'
                        value: -3
                    }
                }
            }
            layer {
                type: 'SoftmaxWithLoss'
                name: 'loss'
                bottom: 'ip'
                bottom: 'label'
                top: 'loss'
            }""")
    f.close()
    return f.name

class TestNetPyCuda(unittest.TestCase):

    def _getTargetClass(self):
            from caffe_facade import Net
            return Net

    def _makeOne(self, *args, **kw):
            return self._getTargetClass()(*args, **kw)

    def setUp(self):
        import caffe_facade
        import numpy as np
        import os

        caffe_facade.set_mode_gpu()
        caffe_facade.set_device(0)
        self.num_output = 13
        net_file = simple_net_file(self.num_output)
        self.net = self._makeOne(net_file, caffe_facade.TRAIN)
        self.net.blobs['label'].data[...] = np.random.randint(self.num_output, size = self.net.blobs['label'].data.shape)
        os.remove(net_file)

    def test_blobs_data(self):
        """
        Test that params_data are the same as blobs data
        """
        import numpy as np

        blobs = list()
        for (blob_name, blob)  in self.net.params.items():
            blobs.append(blob[0].data)
            blobs.append(blob[1].data)

        for (blob, params_data) in zip(blobs, self.net.params_data):
            assert np.allclose(blob, params_data.reshape(blob.shape))

    def test_blobs_diff(self):
        """
        Test that params_diff are the same as blobs diff
        """
        import numpy as np

        blobs = list()
        for (blob_name, blob)  in self.net.params.items():
            blobs.append(blob[0].diff)
            blobs.append(blob[1].diff)

        for (blob, params_diff) in zip(blobs, self.net.params_diff):
            assert np.allclose(blob, params_diff.reshape(blob.shape))

    def test_blob_data_to_gpuarray(self):
        """
        Test data_as_pycuda_gpuarray works for all blobs
        """
        from caffe_facade import pycuda_util

        with pycuda_util.caffe_cuda_context():
            for (blob_name, blob)  in self.net.params.items():
                blob[0].data_as_pycuda_gpuarray()
                blob[1].data_as_pycuda_gpuarray()

    def test_blob_diff_to_gpuarray(self):
        """
        Test diff_as_pycuda_gpuarray works for all blobs
        """
        from caffe_facade import pycuda_util

        with pycuda_util.caffe_cuda_context():
            for (blob_name, blob)  in self.net.params.items():
                blob[0].diff_as_pycuda_gpuarray()
                blob[1].diff_as_pycuda_gpuarray()

    def test_update_data(self):
        """
        Test update data for blobs
        """

        import scikits.cuda.cublas as cublas
        import pycuda.gpuarray as gpuarray
        import copy
        import caffe_facade
        from caffe_facade import pycuda_util
        import numpy as np

        blobs = list()
        for (blob_name, blob)  in self.net.params.items():
            blobs.append(blob[0])
            blobs.append(blob[1])

        mult = 0.0001
        blobs_update_cpu = [np.random.rand(*blob.data.shape).astype(np.float32) * mult for blob in blobs]
        initial_params_data = copy.deepcopy(self.net.params_data)

        with pycuda_util.caffe_cuda_context():
            h = caffe_facade.cublas_handle()
            blobs_gpu = [blob.data_as_pycuda_gpuarray() for blob in blobs]
            blobs_update_gpu = [gpuarray.to_gpu(blob_update_cpu) for blob_update_cpu in blobs_update_cpu]

            for (blob_gpu, blob_update_gpu) in zip(blobs_gpu, blobs_update_gpu):
                cublas.cublasSaxpy(h, blob_gpu.size, 1.0, blob_update_gpu.gpudata, 1, blob_gpu.gpudata, 1)

            for (blob_gpu, initial_param_data, blob_update_cpu) in zip(blobs_gpu, initial_params_data, blobs_update_cpu):
                assert np.allclose(blob_gpu.get(), initial_param_data.reshape(blob_gpu.shape) + blob_update_cpu)

            params_data = self.net.params_data
            for (blob_gpu, param_data) in zip(blobs_gpu, params_data):
                assert np.allclose(blob_gpu.get(), param_data.reshape(blob_gpu.shape))

    def test_update_diff(self):
        """
        Test update diff for blobs
        """

        import scikits.cuda.cublas as cublas
        import pycuda.gpuarray as gpuarray
        import copy
        import caffe_facade
        from caffe_facade import pycuda_util
        import numpy as np

        blobs = list()
        for (blob_name, blob)  in self.net.params.items():
            blobs.append(blob[0])
            blobs.append(blob[1])

        mult = 0.0001
        blobs_update_cpu = [np.random.rand(*blob.diff.shape).astype(np.float32) * mult for blob in blobs]
        initial_params_diff = copy.deepcopy(self.net.params_diff)

        with pycuda_util.caffe_cuda_context():
            h = caffe_facade.cublas_handle()
            blobs_gpu = [blob.diff_as_pycuda_gpuarray() for blob in blobs]
            blobs_update_gpu = [gpuarray.to_gpu(blob_update_cpu) for blob_update_cpu in blobs_update_cpu]

            for (blob_gpu, blob_update_gpu) in zip(blobs_gpu, blobs_update_gpu):
                cublas.cublasSaxpy(h, blob_gpu.size, 1.0, blob_update_gpu.gpudata, 1, blob_gpu.gpudata, 1)

            for (blob_gpu, initial_param_diff, blob_update_cpu) in zip(blobs_gpu, initial_params_diff, blobs_update_cpu):
                assert np.allclose(blob_gpu.get(), initial_param_diff.reshape(blob_gpu.shape) + blob_update_cpu)

            params_diff = self.net.params_diff
            for (blob_gpu, param_diff) in zip(blobs_gpu, params_diff):
                assert np.allclose(blob_gpu.get(), param_diff.reshape(blob_gpu.shape))
