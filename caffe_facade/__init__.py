from .pycaffe import Net, SGDSolver, NesterovSolver, AdaGradSolver

from ._caffe_facade import (
    set_mode_cpu, set_mode_gpu, set_device, Layer,
    check_mode_cpu, check_mode_gpu,
    Blob
)
from .proto.caffe_pb2 import TRAIN, TEST
import io
try:
	from ._caffe_facade import get_cuda_num_threads, get_blocks, cublas_handle
except ImportError:
	pass
