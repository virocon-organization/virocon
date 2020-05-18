import unittest
import os

from viroconcom.read_write import read_ecbenchmark_dataset, read_contour, write_contour

class ReadWriteTest(unittest.TestCase):

    def test_read_dataset(self):
        """
        Reads the provided dataset "1year_dataset_A.txt".
        """
        sample_hs, sample_tz, label_hs, label_tz = read_ecbenchmark_dataset()
        self.assertAlmostEqual(sample_hs[0], 0.2845, delta=0.00001)

    def test_read_write_contour(self):
        """
        Read a contour, then writes this contour to a new file.
        """
        folder_name = 'contour-coordinates/'
        file_name_median = 'doe_john_years_25_median.txt'
        (contour_v_median, contour_hs_median) = read_contour(
            folder_name + file_name_median)
        new_file_path = folder_name + 'test_contour.txt'
        write_contour(contour_v_median, contour_hs_median, new_file_path,
                      'Wind speed (m/s)', 'Significant wave height (m)')
        os.remove(new_file_path)
