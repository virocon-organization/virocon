"""
Info about all of noaa data can be found at:
http://www.ndbc.noaa.gov/docs/ndbc_web_data_guide.pdf
What all the values mean:
http://www.ndbc.noaa.gov/measdes.shtml
        WDIR    Wind direction (degrees clockwise from true N).
        WSPD    Wind speed (m/s) averaged over an eight-minute period.
        GST     Peak 5 or 8 second gust speed (m/s).
        WVHT    Significant wave height (meters) is calculated as
                the average of the highest one-third of all of the
                wave heights during the 20-minute sampling period.
        DPD     Dominant wave period (seconds) is the period with the maximum wave energy.
        APD     Average wave period (seconds) of all waves during the 20-minute period.
        MWD     The direction from which the waves at the dominant period (DPD) are coming
                (degrees clockwise from true N).
        PRES    Sea level pressure (hPa).
        ATMP    Air temperature (Celsius).
        WTMP    Sea surface temperature (Celsius).
        DEWP    Dewpoint temperature.
        VIS     Station visibility (nautical miles).
        PTDY    Pressure Tendency.
        TIDE    The water level in feet above or below Mean Lower Low Water (MLLW).
"""
import datetime
import numpy as np
import pandas as pd
import urllib.request
from sqlalchemy import create_engine


class HistoricData:

    def __init__(self):
        None

    def get_data(self, buoy, date):
        """
        Parameters
        ----------
        date : str
            Form: "year-month-day/to/year-month-day"
            Gives you data from this time period.
        buoy : int
            Buoy number from ndbc.

        Returns
        -------
        df : pandas dataframe
            Containing the data.
        """

        year_start = int(date[0:4])
        month_start = date[5:7]
        day_start = date[8:10]

        year_stop = int(date[14:18])
        month_stop = date[19:21]
        day_stop = date[22:24]
        year_range = (year_start, year_stop)
        df = self.get_year_range(buoy, year_range)

        while str(df.index[0])[0:10] != str(year_start) + "-" + str(month_start) + "-" + str(day_start):
            df = df.drop(df.index[0], axis=0)

        while str(df.index[-1])[0:10] != str(year_stop) + "-" + str(month_stop) + "-" + str(day_stop):
            df = df.drop(df.index[-1], axis=0)

        return df

    def get_year(self, buoy, year):
        """
        Parameters
        ----------
        year : int
            Year from which the data comes from.
        buoy : int
            Buoy number from ndbc.

        Returns
        -------
        df : pandas dataframe
            Containing the data.
        """

        link = 'http://www.ndbc.noaa.gov/view_text_file.php?filename='
        link += '{}h{}.txt.gz&dir=data/historical/'.format(buoy, year)
        link = link + 'stdmet/'

        try:
            df = pd.read_csv(link, header=0, delim_whitespace=True, dtype=object, na_values={99, 999, 9999, 99., 999.,
                                                                                             9999.})
        except:
            return Warning(print('You are trying to get data that does not exists or is not usable from buoy: '
                                 + str(buoy) + ' in year ' + str(year)
                                 + '. Please try a different year or year range without year: ' + str(year)))

        # 2007 and on format.
        if df.iloc[0, 0] == '#yr':
            df = df.rename(columns={'#YY': 'YY'})  # Get rid of hash.
            # Make the indices.
            df.drop(0, inplace=True)  # First row is units, so drop them.
            d = df.YY + ' ' + df.MM + ' ' + df.DD + ' ' + df.hh + ' ' + df.mm
            ind = pd.to_datetime(d, format="%Y %m %d %H %M")
            df.index = ind
            # Drop useless columns and rename the ones we want.
            df.drop(['YY', 'MM', 'DD', 'hh', 'mm'], axis=1, inplace=True)
            df.columns = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD',
                          'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']

        # Before 2006 to 2000.
        else:
            date_str = df.YYYY + ' ' + df.MM + ' ' + df.DD + ' ' + df.hh
            ind = pd.to_datetime(date_str, format="%Y %m %d %H")
            df.index = ind
            # Some data has a minute column. Some doesn't.
            if 'mm' in df.columns:
                df.drop(['YYYY', 'MM', 'DD', 'hh', 'mm'], axis=1, inplace=True)
            else:
                df.drop(['YYYY', 'MM', 'DD', 'hh'], axis=1, inplace=True)
            df.columns = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD',
                          'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']
        # All data should be floats.
        df = df.astype('float')
        return df

    def get_year_range(self, buoy, year_range):
        """
        Parameters
        ----------
        buoy : int
            Buoy, which the data comes from.
        year_range : tuple
            From year to year.
        Returns
        -------
        df : pandas dataframe
            Contains all the data from all the years that were specified
            in year_range.
        """

        start, stop = year_range
        df = pd.DataFrame()  # initialize empty df
        for i in range(start, stop + 1):
            data = self.get_year(buoy, i)
            new_df = data
            df = df.append(new_df)
        return df
