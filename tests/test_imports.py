#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test class for module imports

Author:  mish-mosh
-------

"""

import unittest
import os

from urllib.error import HTTPError
import warnings

import pandas as pd
from pandas.util.testing import assert_frame_equal

from .context import viroconcom
from viroconcom.imports import NDBCImport


_here = os.path.dirname(__file__)
testFiles_path = os.path.abspath(os.path.join(_here, "testfiles"))


class NDBCImportTest(unittest.TestCase):
    # ToDo: constructor test method

    def test_get_virocon_data(self):
        """
        Tests the method get_virocon_data on three cases (see below)

        """

        # ## Test case1: existing single year(2005) (untill 2007) data of a buoy: 41035
        # test dataFrame
        case1_test_NDBC = NDBCImport(41035)
        case1_test_df = case1_test_NDBC.get_virocon_data(2005)

        # comparison dataFrame
        case1_compare_df = pd.read_csv(testFiles_path + "/NDBCimports_single_case1_test_df41035_2005.csv")

        # assertion
        assert_frame_equal(case1_test_df, case1_compare_df)

        # ## Test case2: existing single year(2012) (after 2007) data of a buoy: 41002
        # test dataFrame
        case2_test_NDBC = NDBCImport(41002)
        case2_test_df = case2_test_NDBC.get_virocon_data(2012)

        # comparison dataFrame
        case2_compare_df = pd.read_csv(testFiles_path + "/NDBCimports_single_case2_test_df41002_2012.csv")

        # assertion
        assert_frame_equal(case2_test_df, case2_compare_df)

        # ## Test case3: existing but corrupted data single for year(2011) and a buoy: 41060
        # test dataFrame
        case3_test_NDBC = NDBCImport(41060)
        with warnings.catch_warnings(record=True) as w:
            case3_test_df = case3_test_NDBC.get_virocon_data(2011)
            assert issubclass(w[-1].category, UserWarning)
            assert "corrupted data" in str(w[-1].message)
        self.assertTrue(case3_test_df.empty)

        # ## Test case4: non existing single year (2016) and a buoy: 41058
        case4_test_NDBC = NDBCImport(41058)
        self.assertRaises(HTTPError, case4_test_NDBC.get_virocon_data, 2016,)

    def test_get_virocon_data_range(self):
        """"
            Tests the method get_virocon_data_range on two cases (see below)
        """

        # ## Test case1: existing year range (2007,2011) of a buoy: 41043
        # Test dataFrame
        case1_test_NDBC = NDBCImport(41043)
        case1_test_df = case1_test_NDBC.get_virocon_data_range((2007, 2011))

        # comparison dataFrame
        case1_compare_df = pd.read_csv(testFiles_path + "/NDBCimports_range_test_df41043_2007_2011.csv")

        # assertion
        assert_frame_equal(case1_test_df, case1_compare_df)


if __name__ == '__main__':
    unittest.main()
