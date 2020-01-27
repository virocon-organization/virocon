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

        test_func = FunctionParam('power3', 0, 1, 0)
        self.assertEqual(test_func._value(0), 1)


    def test_FunctionParam_exp3(self):
        """
        tests if function exp3 calculates the correct value.
        """

        test_func = FunctionParam('exp3', 1, 1, 0)
        self.assertEqual(test_func._value(0), 2)

    def test_FunctionParam_lnsquare2(self):
        """
        Tests if function lnsquare2 calculates the correct value.
        """

        test_func = FunctionParam('lnsquare2', 1, 1, None)
        self.assertEqual(test_func._value(0), 0)
        self.assertEqual(test_func.func_name, 'lnsquare2')

    def test_FunctionParam_powerdecrease3(self):
        """
        Tests if function powerdecrease3 calculates the correct value.
        """

        test_func = FunctionParam('powerdecrease3', 1, 2, 2)
        self.assertEqual(test_func._value(0), 1.25)
        self.assertEqual(test_func.func_name, 'powerdecrease3')

    def test_FunctionParam_asymdecrease3(self):
        """
        Tests if function asymdecrease3 calculates the correct value.
        """

        test_func = FunctionParam('asymdecrease3', 1, 4, 2)
        self.assertEqual(test_func._value(0), 1.125)
        self.assertEqual(test_func.func_name, 'asymdecrease3')

    def test_FunctionParam_logistics4(self):
        """
        Tests if function logistics4 calculates the correct value.
        """

        test_func = FunctionParam('logistics4', 1, 2, 3, 4)
        self.assertAlmostEqual(test_func._value(0), 1, delta=0.001)
        self.assertAlmostEqual(test_func._value(10), 3, delta=0.001)
        self.assertEqual(test_func.func_name, 'logistics4')

    def test_FunctionParam_alpha3(self):
        """
        Tests if function alpha3 calculates the correct value.
        """

        # Use the function presented in 'Global hierachical models ...' for dataset D.
        test_func = FunctionParam('alpha3', 0.394, 0.0178, 1.88, C1=0.582, C2=1.90, C3=0.248, C4=8.49)
        self.assertAlmostEqual(test_func._value(0), 0.2, delta=0.2)
        self.assertAlmostEqual(test_func._value(10), 1, delta=0.3)
        self.assertAlmostEqual(test_func._value(20), 4, delta=0.5)
        self.assertEqual(test_func.func_name, 'alpha3')
    def test_FunctionParam_unknown(self):
        """
        tests if the right exception appears when trying to create a non existent
        function
        """

        with self.assertRaises(ValueError):
            FunctionParam('linear', 2.5, 1.0, 0.5)


    def test_Wrapper(self):
        """
        tests if a wrapper object gives the same result as the class Wrapper()
        calculates
        """

        test_func = FunctionParam('power3', 0.5, 1.0, 0.0, wrapper=np.exp)
        test_func2 = FunctionParam('power3', 0.5, 1.0, 0.0, wrapper=Wrapper(np.exp))

        self.assertEqual(test_func._value(9), test_func2._value(9))


if __name__ == '__main__':
    unittest.main()