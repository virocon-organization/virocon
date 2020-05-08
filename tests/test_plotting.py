import unittest

from viroconcom.read_write import read_dataset

class ReadWriteTest(unittest.TestCase):

    def test_read_dataset(self):
        """
        Reads the provided dataset.
        """
        sample_hs, sample_tz, label_hs, label_tz = read_dataset()
        self.assertAlmostEqual(sample_hs[0], 0.2845, delta=0.00001)
