#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Imports data.
"""

import pandas as pd
from scipy import stats
import numpy as np

from urllib.error import HTTPError

import warnings
warnings.simplefilter("always", UserWarning)

__all__ = ["NDBCImport"]


class NDBCImport:
    """
    Imports and holds Historical Standard Meteorological Data from NDBC (see below)

    Example
    -------

    >>> from viroconcom.imports import NDBCImport
    >>> import matplotlib.pyplot as plt
    >>> myNDBC = NDBCImport(41002)
    >>> df = myNDBC.get_virocon_data_range(year_range=(2015, 2018))
    >>> fig, ax = plt.subplots(3, sharex=True)
    >>> df.WSPD.plot(ax=ax[0])
    >>> ax[0].set_ylabel('Wind speed (m/s)', fontsize=8)
    >>> df.WVHT.plot(ax=ax[1])
    >>> ax[1].set_ylabel('Wave hight (m)', fontsize=8)
    >>> df.APD.plot(ax=ax[2])
    >>> ax[2].set_ylabel('Average Period (sec)', fontsize=8)
    >>> ax[2].set_xlabel('')
    >>> fig.suptitle('station: {}, year: {}'.format(myNDBC.buoy,\
                        myNDBC.year_range), fontsize=12)
    >>> plt.show()

    """

    def __init__(self, buoy):
        """

        Parameters
        ----------
        buoy : int
            The buoy (station id)

        """

        self.buoy = buoy

    def get_virocon_data(self, year):
        """
        Gets the Historical Standard Meteorological Data for a specific buoy and year, proceeds it to get only
        three columns (see Notes) and eliminates the outliers.

        Parameters
        ----------
        year : int
            The year in which the data need to be imported

        Notes
        -----
        The fetched data values are listed as follow:

        WSPD:    Wind speed (m/s) averaged over an eight-minute period

        WVHT:    Significant wave height (meters) is calculated as
                the average of the highest one-third of all of the
                wave heights during the 20-minute sampling period.

        APD:     Average wave period (seconds) of all waves during the 20-minute period.

        Raises
        ------
        HTTPError
            If a given buoy was not found or the given year for a given buoy not listed.

        Returns
        -------
        pandas dataframe
            The fetched data.
        """

        link = 'http://www.ndbc.noaa.gov/view_text_file.php?filename='
        link += '{}h{}.txt.gz&dir=data/historical/'.format(self.buoy, year)
        link += 'stdmet/'

        # importing into a dataframe
        try:
            if year >= 2007:
                # skipping second row (first without header) since it is a unit row (2007 and on format)
                df = pd.read_csv(link, header=0, skiprows=[1], delim_whitespace=True, usecols=["WSPD", "WVHT", "APD"],
                                 dtype={"WSPD": "float", "WVHT": "float", "APD": "float"},
                                 na_values=[99, 999, 9999, 99., 999., 9999.])
            else:
                df = pd.read_csv(link, header=0, delim_whitespace=True, usecols=["WSPD", "WVHT", "APD"],
                                 dtype={"WSPD": "float", "WVHT": "float", "APD": "float"},
                                 na_values=[99, 999, 9999, 99., 999., 9999.])
        except HTTPError as e:
            err_msg = "Could not find data for buoy: {} and year: {}!".format(self.buoy, year)
            raise HTTPError(code=e.code, msg=err_msg, hdrs=e.hdrs, fp=e.fp, url=e.url)

        # drop NANs
        df.dropna(inplace=True)

        # handling corrupted data
        if df.empty:
            warnings.warn(f"Empty data frame due to corrupted data! buoy: {self.buoy} - "
                          f"year: {year}", UserWarning)
            return df

        # eliminate the outliers (where the absolute value of Z-Score more than 3)
        df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

        df.reset_index(drop=True, inplace=True)

        return df

    def get_virocon_data_range(self, year_range):
        """
        Gets the Historical Standard Meteorological Data for a specific buoy and year range. See get_virocon_data.

        Parameters
        ----------
        year_range: (int,int)
            The year range in which the data need to be imported

        Raises
        ------
        UserWarning
            If for a given year in a given year range for a given buoy no data listed.

        Returns
        -------
        pandas dataframe
            The fetched data.
        """

        start, end = year_range

        df = pd.DataFrame()
        for current_year in range(start, end+1):
            try:
                df = df.append(self.get_virocon_data(current_year), ignore_index=True)
            except HTTPError as e:
                warnings.warn(e.msg)

        return df
