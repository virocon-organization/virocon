#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA implementation for ViroconCom.

Author:  mish-mosh
"""

import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


__all__ = ["PCAFit"]


class PCAFit(PCA):
    """
    A PCA fitting class, based on the Scikit-Learn's PCA (see below)

    Example
    -------

    >>> import pandas as pd
    >>>
    >>> # read file
    >>> df = pd.read_csv('https://raw.githubusercontent.com/ec-benchmark-organizers/ec-benchmark/master/datasets/A.txt',
    >>>                     sep = ';', usecols=[1,2])
    >>> # initializing the model
    >>> my_pca = PCAFit(df.values)
    >>> ## plotting the original data with original contour
    >>>
    >>> from viroconcom.fitting import Fit
    >>> from viroconcom.contours import IFormContour
    >>>
    >>> dist_description_zup = {'name': 'Lognormal_SigmaMu', 'sigma': (0.00, 0.308, -0.250), 'mu': (1.47, 0.214, 0.641)}
    >>> dist_description_swh = {'name': 'Weibull', 'dependency': (None, None, None), 'number_of_intervals': 2}
    >>>
    >>> # columns to plot and fit
    >>> col1 = df.values.transpose()[1].tolist()
    >>> col2 = df.values.transpose()[0].tolist()
    >>>
    >>> # fitting
    >>> my_fit = Fit((col1, col2), (dist_description_zup, dist_description_swh))
    >>>
    >>> # iform contour
    >>> iform_contour_original = IFormContour(my_fit.mul_var_dist, 10, 5, 500)
    >>>
    >>> # Fitting the Contour with PCA
    >>> contour_fitted = my_pca.get_fitted_contour(iform_contour_original.coordinates[0])
    >>>
    >>> ## plotting the original data with PCA-fitted contour
    >>>
    >>> import matplotlib.pyplot as plt
    >>>
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.scatter(col1, col2, label='sample')
    >>> ax.plot(contour_fitted[0], contour_fitted[1], '-k', label='IForm - contour')
    >>>
    >>> ax.set_xlabel('zero-up-crossing period (s)')
    >>> ax.set_ylabel('significant wave height (m)')
    >>> ax.set_title('the original data with PCA-fitted contour - NDBC - Station: 44007 for 10 years')
    >>>
    >>> plt.legend()
    >>> plt.show()
    """

    def __init__(self, dec_data, **kwargs):
        """

        Parameters
        ----------
        dec_data : ndarray
            data to decompose (with PCA)

        """
        super(PCAFit, self).__init__(n_components=dec_data.shape[1], whiten=True, **kwargs)

        # Standardizing the data
        x = StandardScaler().fit_transform(dec_data)

        # Fitting the model due to the data and transforming the data
        self.fit(x)

    def get_fitted_contour(self, coords):
        """

        :param coords: the coordinates of the contour needed to be fitted. (ndarray/list)
        :return: the fitted contour as ndarray

        """

        # transform the contour
        contour_transposed = np.vstack(coords).transpose()
        contour_transformed = self.transform(contour_transposed)

        # inverse-transform the contour
        contour_fitted = self.inverse_transform(contour_transformed).transpose()
        return contour_fitted
