#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Imports data.
"""

import pandas as pd
from scipy import stats
import numpy as np

from urllib.error import HTTPError

__all__ = ["NDBCImport"]


class NDBCImport:
    """
    Imports and holds Historical Standard Meteorological Data from NDBC (see below)

    Example
    -------

    >>> from viroconcom.imports import NDBCImport
    >>> import matplotlib.pyplot as plt
    >>> myNDBC = NDBCImport(41002, year_range=(2015, 2018))
    >>> myNDBC.write_to_csv('C:')
    >>> fig, ax = plt.subplots(3, sharex=True)
    >>> myNDBC.virocon_data.WSPD.plot(ax=ax[0])
    >>> ax[0].set_ylabel('Wind speed (m/s)', fontsize=8)
    >>> myNDBC.virocon_data.WVHT.plot(ax=ax[1])
    >>> ax[1].set_ylabel('Wave hight (m)', fontsize=8)
    >>> myNDBC.virocon_data.APD.plot(ax=ax[2])
    >>> ax[2].set_ylabel('Average Period (sec)', fontsize=8)
    >>> ax[2].set_xlabel('')
    >>> fig.suptitle('station: {}, year: {}'.format(myNDBC.buoy,\
                        myNDBC.year_range), fontsize=12)
    >>> plt.show()

    """

    def __init__(self, buoy, year=None, year_range=None):
        """

        Parameters
        ----------
        buoy : int
            The buoy (station id)
        year : int
            The year in which the data need to be imported
        year_range: tuple
            The year range in which the data need to be imported

        Raises
        ------
        HTTPError
            If a given buoy was not found or the given year (thus year range also) for a given buoy not listed.

        ValueError
            If year and year_range both were None or contain values at the same time
        """

        self.buoy = buoy
        self.year = year
        self.year_range = year_range

        if year is not None and year_range is None:
            self.virocon_data = self.get_virocon_data()
        elif year is None and year_range is not None:
            self.virocon_data = self.get_virocon_data_range()
        elif year is None and year_range is None:
            raise ValueError("year and year range can not be None at the same time!")
        else:
            raise ValueError("Either year or year range can be submitted!")

    def get_virocon_data(self):
        """
        Gets the Historical Standard Meteorological Data for a specific buoy and year, proceeds it to get only three columns (see
        Notes) and eliminates the outliers.

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
        link += '{}h{}.txt.gz&dir=data/historical/'.format(self.buoy, self.year)
        link += 'stdmet/'

        # importing into a dataframe
        try:
            if self.year >= 2007:
                # skipping second row (first without header) since it is a unit row (2007 and on format)
                df = pd.read_csv(link, header=0, skiprows=[1], delim_whitespace=True, usecols=["WSPD", "WVHT", "APD"],
                                 dtype={"WSPD": "float", "WVHT": "float", "APD": "float"},
                                 na_values=[99, 999, 9999, 99., 999., 9999.])
            else:
                df = pd.read_csv(link, header=0, delim_whitespace=True, usecols=["WSPD", "WVHT", "APD"],
                                 dtype={"WSPD": "float", "WVHT": "float", "APD": "float"},
                                 na_values=[99, 999, 9999, 99., 999., 9999.])
        except HTTPError as e:
            err_msg = "Could not find data for buoy: {} and year: {}!".format(self.buoy, self.year)
            raise HTTPError(code=e.code, msg=err_msg, hdrs=e.hdrs, fp=e.fp, url=e.url)

        # drop NANs
        df.dropna(inplace=True)

        # eliminate the outliers (where the absolute value of Z-Score more than 3)
        df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
        df.reset_index(drop=True, inplace=True)

        return df

    def get_virocon_data_range(self):
        """
        Gets the Historical Standard Meteorological Data for a specific buoy and year range. See get_virocon_data.

        Raises
        ------
        HTTPError
            If a given buoy was not found or a year in a given year range for a given buoy not listed.

        Returns
        -------
        pandas dataframe
            The fetched data.
        """

        start, end = self.year_range
        df = pd.DataFrame()

        for current_year in range(start, end+1):
            self.year = current_year
            df = df.append([self.get_virocon_data()])

        df.reset_index(drop=True, inplace=True)

        return df

    def write_to_csv(self, path):
        """
        Writes the fetched data to a csv file.

        Parameters
        ----------
        path: str
            The path in which the csv file should be saved.
        """

        csv_path = path

        if self.year_range is not None:
            csv_path += '/NDBC_Station_{}_{}.csv'.format(self.buoy, self.year_range)
        else:
            csv_path += '/NDBC_Station_{}_{}.csv'.format(self.buoy, self.year)

        self.virocon_data.to_csv(path_or_buf=csv_path, index=False)
