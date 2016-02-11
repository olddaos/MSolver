import unittest
import tempfile
import os
import numpy as np

import caffe_facade

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

def simple_solver_file(net_filename):
    import tempfile

    f = tempfile.NamedTemporaryFile(delete=False)
    f.write("""net: '""" + net_filename + """'
            test_iter: 10 test_interval: 10 base_lr: 0.01 delta: 0.0001
            weight_decay: 0.0005 lr_policy: 'inv' gamma: 0.0001 power: 0.75
            display: 5 max_iter: 100 snapshot: 10 snapshot_prefix: "test" snapshot_after_train: false""")
    f.close()
    return f.name

class TestAdaGradSolverCPU(unittest.TestCase):

    def _getTargetClass(self):
            from caffe_facade import AdaGradSolver
            return AdaGradSolver

    def _makeOne(self, *args, **kw):
            return self._getTargetClass()(*args, **kw)

    def setUp(self):
        import caffe_facade
        import numpy as np
        import os

        caffe_facade.set_mode_cpu()
        self.num_output = 13
        net_filename = simple_net_file(self.num_output)
        solver_filename = simple_solver_file(net_filename)
        self.solver = self._makeOne(solver_filename)

        self.solver.net.blobs['label'].data[...] = np.random.randint(self.num_output, size = self.solver.net.blobs['label'].data.shape)
        os.remove(net_filename)
        os.remove(solver_filename)

    def test_forward_backward(self):
        """
        Test forward_backward method
        """
        self.solver.forward_backward()

    def test_test(self):
        """
        Test test method
        """
        self.solver.test()

    def test_compute_update_value(self):
        """
        Test compute_update_value method
        """
        self.solver.compute_update_value()

    def test_update(self):
        """
        Test update method
        """
        self.solver.update()

    def test_snapshot(self):
        """
        Test snapshot method
        """
        import os

        self.solver.snapshot()
        os.remove("test_iter_1.solverstate")
        os.remove("test_iter_1.caffemodel")

    def test_calculate_train_info(self):
        """
        Test calculate_train_info method
        """
        self.solver.forward_backward()
        self.solver.calculate_train_info()

    def test_output(self):
        """
        Test output methods
        """
        self.solver.output_train_info()
        self.solver.output_train_loss()
        self.solver.output_learning_rate()
        self.solver.output_finish()

    def test_clear_history(self):
        """
        Test clear_history method
        """
        self.solver.clear_history()

    def test_get_learning_rate(self):
        """
        Test get_learning method
        """
        self.assertAlmostEqual(self.solver.get_learning_rate(), 0.01)

    def test_max_iter(self):
        """
        Test max_iter method
        """
        self.assertEqual(self.solver.max_iter, 100)

    def test_test_interval(self):
        """
        Test test_interval method
        """
        self.assertEqual(self.solver.test_interval, 10)

    def test_snapshot_interval(self):
        """
        Test snapshot_interval method
        """
        self.assertEqual(self.solver.snapshot_interval, 10)

    def test_display(self):
        """
        Test display method
        """
        self.assertEqual(self.solver.display, 5)

    def test_iter(self):
        """
        Test iter property
        """
        self.assertEqual(self.solver.iter, 0)
        self.solver.iter = 20
        self.assertEqual(self.solver.iter, 20)

    def test_train_loss(self):
        """
        Test train_loss property
        """
        self.solver.train_loss = 1
        self.assertAlmostEqual(self.solver.train_loss, 1)

    def test_train_info(self):
        """
        Test train_info property
        """
        self.assertEqual(self.solver.train_info, "")
        self.solver.train_info = "Train Loss"
        self.assertEqual(self.solver.train_info, "Train Loss")

    def test_solve(self):
        """
        Test solve method
        """
        import os

        self.assertEqual(self.solver.iter, 0)
        self.solver.solve()
        self.assertEqual(self.solver.iter, 100)
        for iter in range(10, 101, 10):
            os.remove("test_iter_" + str(iter) + ".solverstate")
            os.remove("test_iter_" + str(iter) + ".caffemodel")

class TestAdaGradSolverGPU(unittest.TestCase):

    def _getTargetClass(self):
            from caffe_facade import AdaGradSolver
            return AdaGradSolver

    def _makeOne(self, *args, **kw):
            return self._getTargetClass()(*args, **kw)

    def setUp(self):
        import caffe_facade
        import numpy as np
        import os

        caffe_facade.set_mode_gpu()
        caffe_facade.set_device(0)
        self.num_output = 13
        net_filename = simple_net_file(self.num_output)
        solver_filename = simple_solver_file(net_filename)
        self.solver = self._makeOne(solver_filename)

        # fill in valid labels
        self.solver.net.blobs['label'].data[...] = np.random.randint(self.num_output, size = self.solver.net.blobs['label'].data.shape)
        os.remove(net_filename)
        os.remove(solver_filename)

    def test_forward_backward(self):
        """
        Test forward_backward method
        """
        self.solver.forward_backward()

    def test_test(self):
        """
        Test test method
        """
        self.solver.test()

    def test_compute_update_value(self):
        """
        Test compute_update_value method
        """
        self.solver.compute_update_value()

    def test_update(self):
        """
        Test update method
        """
        self.solver.update()

    def test_snapshot(self):
        """
        Test snapshot method
        """
        import os

        self.solver.snapshot()
        os.remove("test_iter_1.solverstate")
        os.remove("test_iter_1.caffemodel")

    def test_clear_history(self):
        """
        Test clear_history method
        """
        self.solver.clear_history()

    def test_calculate_train_info(self):
        """
        Test calculate_train_info method
        """
        self.solver.forward_backward()
        self.solver.calculate_train_info()

    def test_solve(self):
        """
        Test solve method
        """
        import os

        self.assertEqual(self.solver.iter, 0)
        self.solver.solve()
        self.assertEqual(self.solver.iter, 100)
        for iter in range(10, 101, 10):
            os.remove("test_iter_" + str(iter) + ".solverstate")
            os.remove("test_iter_" + str(iter) + ".caffemodel")
