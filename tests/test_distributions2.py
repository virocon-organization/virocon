import pytest

import numpy as np
import scipy.stats as sts

from .context import viroconcom
from viroconcom.distributions import (WeibullDistribution, NormalDistribution,
                                      LognormalDistribution)
from viroconcom.params import ConstantParam

# Weibull tests
@pytest.fixture(params=[1, 5, 100])
def weibull_shape(request):
    return ConstantParam(request.param)

@pytest.fixture(params=[0, 1, 100])
def weibull_loc(request):
    return ConstantParam(request.param)

@pytest.fixture(params=[1, 5, 100])
def weibull_scale(request):
    return ConstantParam(request.param)

@pytest.fixture(params=[100, 1000, 5000])
def weibull_number(request):
    return request.param


def test_weibull_cdf(weibull_shape, weibull_loc, weibull_scale):
    x = np.linspace(0, 20)
    ref_cdf = sts.weibull_min.cdf(x, weibull_shape(None), weibull_loc(None), weibull_scale(None))
    dist = WeibullDistribution(weibull_shape, weibull_loc, weibull_scale)
    my_cdf = dist.cdf(x, x, (None, None, None))
    assert np.allclose(ref_cdf, my_cdf)

def test_weibull_i_cdf(weibull_shape, weibull_loc, weibull_scale):
    x = np.linspace(0, 1)
    ref_cdf = sts.weibull_min.ppf(x, weibull_shape(None), weibull_loc(None), weibull_scale(None))
    dist = WeibullDistribution(weibull_shape, weibull_loc, weibull_scale)
    my_cdf = dist.i_cdf(x, x, (None, None, None))
    assert np.allclose(ref_cdf, my_cdf)

def test_weibull_draw_sample(weibull_number, weibull_shape, weibull_loc, weibull_scale):
    ref_points = weibull_number
    dist = WeibullDistribution(weibull_shape, weibull_loc, weibull_scale)
    my_points = dist.draw_sample(weibull_number)
    my_points = my_points.size
    assert ref_points == my_points

@pytest.fixture(params=["shape", "loc", "scale"])
def weibull_param_name(request):
    return request.param

def test_weibull_param_out_of_bounds(weibull_param_name):
    dist = WeibullDistribution()
    setattr(dist, weibull_param_name, ConstantParam(-np.inf))
    with pytest.raises(ValueError):
        dist.cdf([0, 100], [0, 100], (None, None, None))

    dist = WeibullDistribution()
    setattr(dist, weibull_param_name, ConstantParam(np.inf))
    with pytest.raises(ValueError):
        dist.cdf([0, 100], [0, 100], (None, None, None))


# Normal tests
@pytest.fixture(params=[0, 1, 100, -10])
def normal_loc(request):
    return ConstantParam(request.param)

@pytest.fixture(params=[1, 5, 100])
def normal_scale(request):
    return ConstantParam(request.param)

def test_normal_cdf(normal_loc, normal_scale):
    x = np.linspace(-20, 20)
    ref_cdf = sts.norm.cdf(x, normal_loc(None), normal_scale(None))
    dist = NormalDistribution(None, normal_loc, normal_scale)
    my_cdf = dist.cdf(x, x, (None, None, None))
    assert np.allclose(ref_cdf, my_cdf)

def test_normal_i_cdf(normal_loc, normal_scale):
    x = np.linspace(0, 1)
    ref_cdf = sts.norm.ppf(x, normal_loc(None), normal_scale(None))
    dist = NormalDistribution(None, normal_loc, normal_scale)
    my_cdf = dist.i_cdf(x, x, (None, None, None))
    assert np.allclose(ref_cdf, my_cdf)

@pytest.fixture(params=["shape", "loc", "scale"])
def normal_param_name(request):
    return request.param

def test_normal_param_out_of_bounds(normal_param_name):
    dist = NormalDistribution()
    setattr(dist, normal_param_name, ConstantParam(-np.inf))
    with pytest.raises(ValueError):
        dist.cdf([0, 100], [0, 100], (None, None, None))

    dist = NormalDistribution()
    setattr(dist, normal_param_name, ConstantParam(np.inf))
    with pytest.raises(ValueError):
        dist.cdf([0, 100], [0, 100], (None, None, None))


# Lognormal tests
@pytest.fixture(params=[1, 5, 100])
def lognormal_shape(request):
    return ConstantParam(request.param)

@pytest.fixture(params=[1, 5, 100])
def lognormal_scale(request):
    return ConstantParam(request.param)

def test_lognormal_cdf(lognormal_shape, lognormal_scale):
    x = np.linspace(0, 20)
    ref_cdf = sts.lognorm.cdf(x, s=lognormal_shape(None), scale=lognormal_scale(None))
    dist = LognormalDistribution(lognormal_shape, None, lognormal_scale)
    my_cdf = dist.cdf(x, x, (None, None, None))
    assert np.allclose(ref_cdf, my_cdf)

def test_lognormal_i_cdf(lognormal_shape, lognormal_scale):
    x = np.linspace(0, 1)
    ref_cdf = sts.lognorm.ppf(x, s=lognormal_shape(None), scale=lognormal_scale(None))
    dist = LognormalDistribution(lognormal_shape, None, lognormal_scale)
    my_cdf = dist.i_cdf(x, x, (None, None, None))
    assert np.allclose(ref_cdf, my_cdf)


@pytest.fixture(params=["shape", "scale"])
def lognormal_param_name(request):
    return request.param

def test_lognormal_param_out_of_bounds(lognormal_param_name):
    dist = LognormalDistribution()
    setattr(dist, lognormal_param_name, ConstantParam(-np.inf))
    with pytest.raises(ValueError):
        dist.cdf([0, 100], [0, 100], (None, None, None))

    dist = LognormalDistribution()
    setattr(dist, lognormal_param_name, ConstantParam(np.inf))
    with pytest.raises(ValueError):
        dist.cdf([0, 100], [0, 100], (None, None, None))

