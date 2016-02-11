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

class TestNet(unittest.TestCase):

    def _getTargetClass(self):
            from caffe_facade import Net
            return Net

    def _makeOne(self, *args, **kw):
            return self._getTargetClass()(*args, **kw)

    def setUp(self):
        import caffe_facade
        import os
        import numpy as np

        self.num_output = 13
        net_file = simple_net_file(self.num_output)
        self.net = self._makeOne(net_file, caffe_facade.TRAIN)
        self.net.blobs['label'].data[...] = np.random.randint(self.num_output, size = self.net.blobs['label'].data.shape)
        os.remove(net_file)

    def test_params_data_shape(self):
        """
        Test params_data shape for initialized net
        """

        self.assertEqual(len(self.net.params_data), 4)
        self.assertEqual(self.net.params_data[0].shape, (11, 2, 2, 2))
        self.assertEqual(self.net.params_data[1].shape, (11, 1, 1, 1))
        self.assertEqual(self.net.params_data[2].shape, (13, 792, 1, 1))
        self.assertEqual(self.net.params_data[3].shape, (13, 1, 1, 1))

    def test_params_diff_shape(self):
        """
        Test params_diff shape for initialized net
        """

        self.assertEqual(len(self.net.params_diff), 4)
        self.assertEqual(self.net.params_diff[0].shape, (11, 2, 2, 2))
        self.assertEqual(self.net.params_diff[1].shape, (11, 1, 1, 1))
        self.assertEqual(self.net.params_diff[2].shape, (13, 792, 1, 1))
        self.assertEqual(self.net.params_diff[3].shape, (13, 1, 1, 1))

    def test_params_items(self):
        """
        Test name and shape of layers for initialized net
        """

        blob_name_conv, blob_conv = self.net.params.items()[0]
        self.assertEqual(blob_name_conv, 'conv')
        self.assertEqual(blob_conv[0].data.shape, (11, 2, 2, 2))
        self.assertEqual(blob_conv[0].diff.shape, (11, 2, 2, 2))
        self.assertEqual(blob_conv[1].data.shape, (11, ))
        self.assertEqual(blob_conv[1].diff.shape, (11, ))

        blob_name_ip, blob_ip = self.net.params.items()[1]
        self.assertEqual(blob_name_ip, 'ip')
        self.assertEqual(blob_ip[0].data.shape, (13, 792))
        self.assertEqual(blob_ip[0].diff.shape, (13, 792))
        self.assertEqual(blob_ip[1].data.shape, (13, ))
        self.assertEqual(blob_ip[1].diff.shape, (13, ))

    def test_params_data_write_read(self):
        """
        Test manipulation with params_data for initialized net
        """
        import numpy as np

        random_data = list()
        for param in self.net.params_data:
            random_data.append(np.random.randn(*param.shape).astype('float32'))
        self.net.params_data = random_data
        for (param, random_arr) in zip(self.net.params_data, random_data):
            self.assertTrue(np.all(random_arr == param))

    def test_params_diff_write_read(self):
        """
        Test manipulation with params_diff for initialized net
        """
        import numpy as np

        random_diff = list()
        for param in self.net.params_diff:
            random_diff.append(np.random.randn(*param.shape).astype('float32'))
        self.net.params_diff = random_diff
        for (param, random_arr) in zip(self.net.params_diff, random_diff):
            self.assertTrue(np.all(random_arr == param))

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

    def test_memory(self):
        """
        Test that holding onto blob data beyond the life of a initialized net is OK
        """

        params = sum(map(list, self.net.params.itervalues()), [])
        blobs = self.net.blobs.values()
        del self.net

        total = 0
        for p in params:
            total += p.data.sum() + p.diff.sum()
        for bl in blobs:
            total += bl.data.sum() + bl.diff.sum()

    def test_forward_backward(self):
        """
        Test forward and backward methods
        """
        self.net.forward()
        self.net.backward()

    def test_inputs_outputs(self):
        """
        Test inputs and outputs of the net
        """
        self.assertEqual(self.net.inputs, [])
        self.assertEqual(self.net.outputs, ['loss'])

    def test_save_and_read(self):
        """
        Test save and read
        """
        import tempfile
        import caffe_facade
        import os

        f = tempfile.NamedTemporaryFile(delete=False)
        f.close()
        self.net.save(f.name)
        net_file = simple_net_file(self.num_output)
        net2 = self._makeOne(net_file, f.name, caffe_facade.TRAIN)
        os.remove(net_file)
        os.remove(f.name)
        for name in self.net.params:
            for i in range(len(self.net.params[name])):
                self.assertEqual(abs(self.net.params[name][i].data
                    - net2.params[name][i].data).sum(), 0)
