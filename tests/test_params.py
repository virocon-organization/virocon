#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:22:42 2017

@author: nb
"""
import unittest

import numpy as np


from .context import viroconcom

from viroconcom.params import ConstantParam, FunctionParam, Wrapper


class ParamsTest(unittest.TestCase):


    def test_ConstantParam(self):
        """
        tests if ConstantParam passes the value it gets on call
        """

        test = ConstantParam(1.471)
        self.assertEqual(test._value(test), 1.471)


    def test_FunctionParam_power3(self):
        """
        tests if function power3 calculates the correct value
        """

        test_func = FunctionParam(0, 1, 0, 'power3')
        self.assertEqual(test_func._value(0), 1)


    def test_FunctionParam_exp3(self):
        """
        tests if function exp3 calculates the correct value
        """

        test_func = FunctionParam(1, 1, 0, 'exp3')
        self.assertEqual(test_func._value(0), 2)

    def test_FunctionParam_lnsquare2(self):
        """
        Tests if function lnsquare2 calculates the correct value
        """

        test_func = FunctionParam(1, 1, None, 'lnsquare2')
        self.assertEqual(test_func._value(0), 0)
        self.assertEqual(test_func.func_name, 'lnsquare2')


    def test_FunctionParam_unknown(self):
        """
        tests if the right exception appears when trying to create a non existent
        function
        """

        with self.assertRaises(ValueError):
            FunctionParam(2.5, 1.0, 0.5, 'linear')


    def test_Wrapper(self):
        """
        tests if a wrapper object gives the same result as the class Wrapper()
        calculates
        """

        test_func = FunctionParam(0.5, 1.0, 0.0, 'power3', wrapper=np.exp)
        test_func2 = FunctionParam(0.5, 1.0, 0.0, 'power3', wrapper=Wrapper(np.exp))

        self.assertEqual(test_func._value(9), test_func2._value(9))


if __name__ == '__main__':
    unittest.main()