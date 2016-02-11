import unittest

class TestModels(unittest.TestCase):
    def _getTargetClass(self):
            from caffe_facade import SGDSolver
            return SGDSolver

    def _makeOne(self, *args, **kw):
            return self._getTargetClass()(*args, **kw)

    def setUp(self):
        import caffe_facade
        caffe_facade.set_mode_gpu()
        caffe_facade.set_device(0)

    def test_cifar10_full(self):
        """
        Test SGDSolver on CIFAR10 for several iterations
        """
        import os

        os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data')))
        solver = self._makeOne('cifar10_full_solver.prototxt')
        solver.init_solve('')
        solver.solve()

    def test_alexnet(self):
        """
        Test AlexNet on ImageNet 1K for several iterations
        """
        import os

        os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data')))
        solver = self._makeOne('alexnet_solver.prototxt')
        solver.init_solve('')
        solver.solve()

    def test_nin(self):
        """
        Test NiN on ImageNet 1K for several iterations
        """
        import os

        os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data')))
        solver = self._makeOne('nin_solver.prototxt')
        solver.init_solve('')
        solver.solve()

    def test_googlenet(self):
        """
        Test GoogLeNet on ImageNet 1K for several iterations
        """
        import os
        
        os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_data')))
        solver = self._makeOne('googlenet_quick_solver.prototxt')
        solver.init_solve('')
        solver.solve()
