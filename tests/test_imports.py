#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test class for module imports

Author:  mish-mosh
-------

"""

import unittest
import os


_here = os.path.dirname(__file__)
testfiles_path = os.path.abspath(os.path.join(_here, "testfiles"))


class NDBCImportTest(unittest.TestCase):
    # ToDo: constructor test method

    def test_buoy_year_existing(self):
        pass


if __name__ == '__main__':
    unittest.main()
