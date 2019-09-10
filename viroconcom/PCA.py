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