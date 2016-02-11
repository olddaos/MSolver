"""
Wrap the internal caffe C++ module (_caffe.so) with a clean, Pythonic
interface.
"""

from collections import OrderedDict
try:
	from itertools import izip_longest
except:
	from itertools import zip_longest as izip_longest
import numpy as np

from ._caffe_facade import Net, SGDSolver, NesterovSolver, AdaGradSolver

# We directly update methods from Net here (rather than using composition or
# inheritance) so that nets created by caffe (e.g., by SGDSolver) will
# automatically have the improved interface.


@property
def _Net_blobs(self):
    """
    An OrderedDict (bottom to top, i.e., input to output) of network
    blobs indexed by name
    """
    return OrderedDict(zip(self._blob_names, self._blobs))


@property
def _Net_params(self):
    """
    An OrderedDict (bottom to top, i.e., input to output) of network
    parameters indexed by name; each is a list of multiple blobs (e.g.,
    weights and biases)
    """
    return OrderedDict([(name, lr.blobs)
                        for name, lr in zip(self._layer_names, self.layers)
                        if len(lr.blobs) > 0])


@property
def _Net_inputs(self):
    return [list(self.blobs.keys())[i] for i in self._inputs]


@property
def _Net_outputs(self):
    return [list(self.blobs.keys())[i] for i in self._outputs]

def _Net_forward(self, blobs=None, start=None, end=None, **kwargs):
    """
    Forward pass: prepare inputs and run the net forward.

    Take
    blobs: list of blobs to return in addition to output blobs.
    kwargs: Keys are input blob names and values are blob ndarrays.
            For formatting inputs for Caffe, see Net.preprocess().
            If None, input is taken from data layers.
    start: optional name of layer at which to begin the forward pass
    end: optional name of layer at which to finish the forward pass (inclusive)

    Give
    outs: {blob name: blob ndarray} dict.
    """
    if blobs is None:
        blobs = []

    if start is not None:
        start_ind = list(self._layer_names).index(start)
    else:
        start_ind = 0

    if end is not None:
        end_ind = list(self._layer_names).index(end)
        outputs = set([end] + blobs)
    else:
        end_ind = len(self.layers) - 1
        outputs = set(self.outputs + blobs)

    if kwargs:
        if set(kwargs.keys()) != set(self.inputs):
            raise Exception('Input blob arguments do not match net inputs.')
        # Set input according to defined shapes and make arrays single and
        # C-contiguous as Caffe expects.
        for in_, blob in kwargs.iteritems():
            if blob.ndim != 4:
                raise Exception('{} blob is not 4-d'.format(in_))
            if blob.shape[0] != self.blobs[in_].num:
                raise Exception('Input is not batch sized')
            self.blobs[in_].data[...] = blob

    self._forward(start_ind, end_ind)

    # Unpack blobs to extract
    return {out: self.blobs[out].data for out in outputs}


def _Net_backward(self, diffs=None, start=None, end=None, **kwargs):
    """
    Backward pass: prepare diffs and run the net backward.

    Take
    diffs: list of diffs to return in addition to bottom diffs.
    kwargs: Keys are output blob names and values are diff ndarrays.
            If None, top diffs are taken from forward loss.
    start: optional name of layer at which to begin the backward pass
    end: optional name of layer at which to finish the backward pass (inclusive)

    Give
    outs: {blob name: diff ndarray} dict.
    """
    if diffs is None:
        diffs = []

    if start is not None:
        start_ind = list(self._layer_names).index(start)
    else:
        start_ind = len(self.layers) - 1

    if end is not None:
        end_ind = list(self._layer_names).index(end)
        outputs = set([end] + diffs)
    else:
        end_ind = 0
        outputs = set(self.inputs + diffs)

    if kwargs:
        if set(kwargs.keys()) != set(self.outputs):
            raise Exception('Top diff arguments do not match net outputs.')
        # Set top diffs according to defined shapes and make arrays single and
        # C-contiguous as Caffe expects.
        for top, diff in kwargs.iteritems():
            if diff.ndim != 4:
                raise Exception('{} diff is not 4-d'.format(top))
            if diff.shape[0] != self.blobs[top].num:
                raise Exception('Diff is not batch sized')
            self.blobs[top].diff[...] = diff

    self._backward(start_ind, end_ind)

    # Unpack diffs to extract
    return {out: self.blobs[out].diff for out in outputs}


# Attach methods to Net.
Net.blobs = _Net_blobs
Net.params = _Net_params
Net.inputs = _Net_inputs
Net.outputs = _Net_outputs
Net.forward = _Net_forward
Net.backward = _Net_backward
