# test the caffe_facade.Blob class
import unittest

class TestBlob(unittest.TestCase):

    def _getTargetClass(self):
        from caffe_facade import Blob
        return Blob

    def _makeOne(self, *args, **kw):
        return self._getTargetClass()(*args, **kw)

    def test_empty_init(self):
        """
        Test empty initialization for Blob class
        """
        blob = self._makeOne([])
        self.assertEqual(blob.shape, ())

    def test_list_init(self):
        """
        Test initialization by the list for Blob class
        """
        blob = self._makeOne([1, 2, 3])
        self.assertEqual(blob.shape, (1, 2, 3))

    def test_tuple_init(self):
        """
        Test initialization by the tuple for Blob class
        """
        blob = self._makeOne((1, 2, 3))
        self.assertEqual(blob.shape, (1, 2, 3))
        blob = self._makeOne(xrange(2, 6))
        self.assertEqual(blob.shape, tuple(xrange(2, 6)))

    def test_set_data(self):
        """
        Test manipulation of the data for Blob class
        """
        import numpy as np

        blob = self._makeOne([1, 2, 3, 4])
        a = np.random.randn(1, 2, 3 ,4)
        blob.data[...] = a
        self.assertTrue(np.all(a.astype('float32') == blob.data))

        blob = self._makeOne((1, 2, 3, 4))
        a = np.random.randn(1, 2, 3, 4)
        blob.data[...] = a
        self.assertTrue(np.all(a.astype('float32') == blob.data))

    def test_set_diff(self):
        """
        Test manipulation of the diff for Blob class
        """
        import numpy as np

        blob = self._makeOne([1, 2, 3, 4])
        a = np.random.randn(1, 2, 3 ,4)
        blob.diff[...] = a
        self.assertTrue(np.all(a.astype('float32') == blob.diff))

        blob = self._makeOne((1, 2, 3, 4))
        a = np.random.randn(1, 2, 3, 4)
        blob.diff[...] = a
        self.assertTrue(np.all(a.astype('float32') == blob.diff))

    def test_num(self):
        """
        Test num property for Blob class
        """
        blob = self._makeOne([1, 2, 3, 4])
        self.assertEqual(blob.num, 1)
        blob = self._makeOne([4, 3, 2, 1])
        self.assertEqual(blob.num, 4)

        blob = self._makeOne((1, 2, 3, 4))
        self.assertEqual(blob.num, 1)
        blob = self._makeOne((4, 3, 2, 1))
        self.assertEqual(blob.num, 4)

    def test_channels(self):
        """
        Test channels property for Blob class
        """

        blob = self._makeOne([1, 2, 3, 4])
        self.assertEqual(blob.channels, 2)
        blob = self._makeOne([4, 3, 2, 1])
        self.assertEqual(blob.channels, 3)

        blob = self._makeOne((1, 2, 3, 4))
        self.assertEqual(blob.channels, 2)
        blob = self._makeOne((4, 3, 2, 1))
        self.assertEqual(blob.channels, 3)

    def test_height(self):
        """
        Test height property for Blob class
        """

        blob = self._makeOne([1, 2, 3, 4])
        self.assertEqual(blob.height, 3)
        blob = self._makeOne([4, 3, 2, 1])
        self.assertEqual(blob.height, 2)

        blob = self._makeOne((1, 2, 3, 4))
        self.assertEqual(blob.height, 3)
        blob = self._makeOne((4, 3, 2, 1))
        self.assertEqual(blob.height, 2)

    def test_width(self):
        """
        Test width property for Blob class
        """

        blob = self._makeOne([1, 2, 3, 4])
        self.assertEqual(blob.width, 4)
        blob = self._makeOne([4, 3, 2, 1])
        self.assertEqual(blob.width, 1)

        blob = self._makeOne((1, 2, 3, 4))
        self.assertEqual(blob.width, 4)
        blob = self._makeOne((4, 3, 2, 1))
        self.assertEqual(blob.width, 1)

    def test_count(self):
        """
        Test count property for Blob class
        """

        blob = self._makeOne([1, 2, 3, 4])
        self.assertEqual(blob.count, 24)
        blob = self._makeOne([4, 3, 2, 1])
        self.assertEqual(blob.count, 24)

        blob = self._makeOne((1, 2, 3, 4))
        self.assertEqual(blob.count, 24)
        blob = self._makeOne((4, 3, 2, 1))
        self.assertEqual(blob.count, 24)

    def test_reshape(self):
        """
        Test reshape method for Blob class
        """

        blob = self._makeOne([1, 2, 3, 4])
        blob.reshape(1, 2, 3, 4)
        self.assertEqual(blob.shape, (1, 2, 3, 4))
        blob.reshape(4, 3, 2, 1)
        self.assertEqual(blob.shape, (4, 3, 2, 1))

        blob = self._makeOne((1, 2, 3, 4))
        blob.reshape(1, 2, 3, 4)
        self.assertEqual(blob.shape, (1, 2, 3, 4))
        blob.reshape(4, 3, 2, 1)
        self.assertEqual(blob.shape, (4, 3, 2, 1))
