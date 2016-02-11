import os
from contextlib import contextmanager

import pycuda.driver as cuda
import pycuda.gpuarray

from caffe_facade import Blob

# Set pycuda array getter to Blob class


def _Blob_data_as_pycuda_gpuarray(self):
    return pycuda.gpuarray.GPUArray(self.shape, 'float32', base=self,
                                    gpudata=self.gpu_data_ptr)


def _Blob_diff_as_pycuda_gpuarray(self):
    return pycuda.gpuarray.GPUArray(self.shape, 'float32', base=self,
                                    gpudata=self.gpu_diff_ptr)
Blob.data_as_pycuda_gpuarray = _Blob_data_as_pycuda_gpuarray
Blob.diff_as_pycuda_gpuarray = _Blob_diff_as_pycuda_gpuarray


def block_and_grid(count):
    import caffe_facade
    d = {}
    d['block'] = (caffe_facade.get_cuda_num_threads(), 1, 1)
    d['grid'] = (caffe_facade.get_blocks(count), 1)
    return d


def get_include_dir():
    path_makefile = os.path.join(os.path.dirname(__file__), 'Makefile.config')
    caffe_dir = filter(lambda x: x.startswith("CAFFE_DIR"), open(path_makefile, 'r'))[0]
    caffe_dir = caffe_dir.split(':=')[-1].rstrip().lstrip()
    caffe_include_dir = os.path.abspath(os.path.join(caffe_dir, 'include'))
    return caffe_include_dir

caffe_include_dirs = [get_include_dir()]


@contextmanager
def caffe_cuda_context():
    """Borrow CUDA context from CAFFE duraing `with` statement. See
        `test_pycuda.py` for usage. We strongly recommend to use this context
        manager if you use PyCuda with CAFFE. Do not use `pycuda.autoinit`.

    NOTE: This context manager uses `cuCtxAttach` and `cuCtxDetach` functions
        (see Driver API in CUDA programing guide, and they are exposed as
        `attach` and `detach` in PyCuda) to get the current CUDA context used
        in CAFFE. Actually these functions are deprecated and the CUDA
        programing guide (v6.5) recommends to use `cuCtxGetCurrent` instead,
        but PyCuda does not expose it to Python I/F. Hence, here we use
        `attach` and `detach` function so far.
    """
    import caffe_facade
    if not caffe_facade.check_mode_gpu():
        raise ValueError(
            "PyCuda cannot be used if Caffe is not in GPU mode.")
    ctx = cuda.Context.attach()
    try:
        yield
    finally:
        ctx.detach()


class CaffeCudaContext:
    """See doc of `caffe_cuda_context`"""
    def __init__(self):
        import caffe_facade
        if not caffe_facade.check_mode_gpu():
            raise ValueError(
                "PyCuda cannot be used if Caffe is not in GPU mode.")
        self.ctx_ = cuda.Context.attach()

    def synchronize(self):
        self.ctx_.synchronize()

    def __del__(self):
        self.ctx_.detach()
