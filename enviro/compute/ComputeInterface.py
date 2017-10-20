import pandas as pd
from enviro.models import MeasureFileManager
from ..models import ParameterModel, DistributionModel, ProbabilisticModel
from .distributions import *
from .contours import *
from .params import *
from .fitting import *
import warnings

class ComputeInterface:
    @staticmethod
    def fit_curves(mfm_item: MeasureFileManager, fit_settings, var_number):
        """
        The method represents the interface to compute to fit measure files.
        :param mfm_item:        measure file item form the MeasureFileModel. 
        :param fit_settings:    the selected fit settings form the user. 
        :param var_number:      number of variables. 
        :return:                the results of the fits ( a lot of arrays). 
        """
        data_path = mfm_item.measure_file.url
        data_path = data_path[1:]
        data = pd.read_csv(data_path, sep=';', header=1).as_matrix()
        dists = []
        dates = []
        for i in range(0, var_number):
            dates.append(data[:, i].tolist())
            if i == 0:
                dists.append(
                    {'name': fit_settings['distribution_%s' % i],
                     'dependency': (None, None, None)})
            else:

                dists.append(
                    {'name': fit_settings['distribution_%s' % i],
                     'dependency': (adjust(fit_settings['shape_dependency_%s' % i][0]),
                                    adjust(fit_settings['location_dependency_%s' % i][0]),
                                    adjust(fit_settings['scale_dependency_%s' % i][0])),
                     'functions': (adjust(fit_settings['shape_dependency_%s' % i][1:]),
                                   adjust(fit_settings['location_dependency_%s' % i][1:]),
                                   adjust(fit_settings['scale_dependency_%s' % i][1:]))})
        fit = Fit(dates, dists, n_steps=int(fit_settings['number_of_intervals']))
        return fit

    @staticmethod
    def iform(probabilistic_model: ProbabilisticModel, return_period, sea_state, n_steps):
        """
        The method represents the interface to compute to calculate IFORM contours.
        :param probabilistic_model: the item which stores the info for a MultivariateDistribution.
        :param return_period:       considered years. 
        :param n_steps:             number of points on the contour.
        :param sea_state:           
        :return:                    a matrix with x and y coordinates which represents the contour.
        """
        mul_dist = setup_mul_dist(probabilistic_model)
        contour = IFormContour(mul_dist, int(return_period), int(sea_state), int(n_steps))
        return contour.coordinates

    @staticmethod
    def hdc(probabilistic_model: ProbabilisticModel, return_period, state_duration, limits, deltas):
        """
        The method represents the interface to compute to calculate HDC contours.
        :param probabilistic_model: the item which stores the info for a MultivariateDistribution.
        :param limits:              limits 
        :param deltas:              deltas 
        :param return_period:       considered years 
        :param state_duration:      
        :return:                    a matrix with x and y coordinates which represents the contour.
        """
        mul_dist = setup_mul_dist(probabilistic_model)
        contour = HighestDensityContour(mul_dist, return_period=return_period, state_duration=state_duration,
                                        limits=limits, deltas=deltas)
        return contour.coordinates


def adjust(var):
    """
    The function adjusts the values from a database model.
    :param var:     variable to adjust
    :return:        the adjusted variable
    """
    if var == 'None' or var == '!':
        return None
    elif var.isdigit():
        return int(var)
    else:
        return var


def setup_mul_dist(probabilistic_model: ProbabilisticModel):
    """
    The function generates a MultivariateDistribution form a ProbabilisticModel item (database).
    :param probabilistic_model: the item which stores the info for a MultivariateDistribution.
    :return:                    MultivariateDistribution 
    """
    distributions_model = DistributionModel.objects.filter(probabilistic_model=probabilistic_model)
    distributions = []
    dependencies = []

    for dist in distributions_model:
        dependency = []
        parameters = []
        parameters_model = ParameterModel.objects.filter(distribution=dist)
        for param in parameters_model:
            dependency.append(adjust(param.dependency))

            if adjust(param.function) is not None:
                parameters.append(FunctionParam(float(param.x0), float(param.x1), float(param.x2), param.function))
            else:
                parameters.append(ConstantParam(float(param.x0)))

        dependencies.append(dependency)

        if dist.distribution == 'Normal':
            distributions.append(NormalDistribution(*parameters))
        elif dist.distribution == 'Weibull':
            distributions.append(WeibullDistribution(*parameters))
        elif dist.distribution == 'Lognormal_2':
            distributions.append(LognormalDistribution(sigma=parameters[0], mu=parameters[2]))
        elif dist.distribution == 'KernelDensity':
            distributions.append(KernelDensityDistribution(*parameters))
        else:
            raise KeyError('{} is not a matching distribution'.format(dist.distribution))

    return MultivariateDistribution(distributions, dependencies)
