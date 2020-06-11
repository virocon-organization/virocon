import pytest

import numpy as np
import datetime

from viroconcom.dataNDBC import NDBC


# NDBC test.
def test_get_data(date="2017-01-01/to/2017-12-31", buoy=41108):
    # Reference numbers from excel calculation.
    ref_average_wvht = 1.0407423375
    ref_average_apd = 4.6799993264
    data = NDBC(buoy).get_data(date)
    average_wvht = np.mean(data.WVHT)
    average_apd = np.mean(data.APD)
    assert np.allclose(ref_average_wvht, average_wvht) and \
           np.allclose(ref_average_apd, average_apd)


def test_get_data2(date="2017-01-14/to/2017-05-22", buoy=41108):
    start = datetime.datetime.strptime(date[:10], '%Y-%m-%d')
    end = datetime.datetime.strptime(date[14:], '%Y-%m-%d')
    days = end - start
    ref_count = int(days.days) * 24 * 2
    count = len(NDBC(buoy).get_data(date).WVHT)
    # print(ref_count, count)
    assert count + 200 >= ref_count >= count - 200


def test_get_data3(date="2003-11-14/to/2004-02-22", buoy=46084):
    data = NDBC(buoy).get_data(date)
    assert len(data) > 0
