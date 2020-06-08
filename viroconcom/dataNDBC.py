"""

Functions to query the NDBC (http://www.ndbc.noaa.gov/).

Info about all of noaa data can be found at:
http://www.ndbc.noaa.gov/docs/ndbc_web_data_guide.pdf

What all the values mean:
http://www.ndbc.noaa.gov/measdes.shtml

"""

import datetime
import numpy as np
import pandas as pd
import urllib.request
from sqlalchemy import create_engine


class HistoricData:

    def __init__(self):
        None

    def get_stand_meteo(self, buoy, year):
        '''
        Standard Meteorological Data. Data header was changed in 2007. Thus
        the need for the if statement below.



        WDIR    Wind direction (degrees clockwise from true N)
        WSPD    Wind speed (m/s) averaged over an eight-minute period
        GST     Peak 5 or 8 second gust speed (m/s)
        WVHT    Significant wave height (meters) is calculated as
                the average of the highest one-third of all of the
                wave heights during the 20-minute sampling period.
        DPD     Dominant wave period (seconds) is the period with the maximum wave energy.
        APD     Average wave period (seconds) of all waves during the 20-minute period.
        MWD     The direction from which the waves at the dominant period (DPD) are coming.
                (degrees clockwise from true N)
        PRES    Sea level pressure (hPa).
        ATMP    Air temperature (Celsius).
        WTMP    Sea surface temperature (Celsius).
        DEWP    Dewpoint temperature
        VIS     Station visibility (nautical miles).
        PTDY    Pressure Tendency
        TIDE    The water level in feet above or below Mean Lower Low Water (MLLW).
        '''

        link = 'http://www.ndbc.noaa.gov/view_text_file.php?filename='
        link += '{}h{}.txt.gz&dir=data/historical/'.format(buoy, year)
        link = link + 'stdmet/'

        # combine the first five date columns YY MM DD hh and make index
        try:
            df = pd.read_csv(link, header=0, delim_whitespace=True, dtype=object, na_values={99, 999, 9999, 99., 999.,
                                                                                             9999.})
        except:
            return Warning(print('You are trying to get data that does not exists or is not usable from buoy: '
                                 + str(buoy) + ' in year ' + str(year)
                                 + '. Please try a different year or year range without year: ' + str(year)))

        # 2007 and on format
        if df.iloc[0, 0] == '#yr':

            df = df.rename(columns={'#YY': 'YY'})  # get rid of hash

            # make the indices

            df.drop(0, inplace=True)  # first row is units, so drop them

            d = df.YY + ' ' + df.MM + ' ' + df.DD + ' ' + df.hh + ' ' + df.mm
            ind = pd.to_datetime(d, format="%Y %m %d %H %M")

            df.index = ind

            # drop useless columns and rename the ones we want
            df.drop(['YY', 'MM', 'DD', 'hh', 'mm'], axis=1, inplace=True)
            df.columns = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD',
                          'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']


        # before 2006 to 2000
        else:
            date_str = df.YYYY + ' ' + df.MM + ' ' + df.DD + ' ' + df.hh

            ind = pd.to_datetime(date_str, format="%Y %m %d %H")

            df.index = ind

            # some data has a minute column. Some doesn't.

            if 'mm' in df.columns:
                df.drop(['YYYY', 'MM', 'DD', 'hh', 'mm'], axis=1, inplace=True)
            else:
                df.drop(['YYYY', 'MM', 'DD', 'hh'], axis=1, inplace=True)

            df.columns = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD',
                          'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']

        # all data should be floats
        df = df.astype('float')
        return df

    def get_all_stand_meteo(self, buoy, year_range):
        """
        Retrieves all the standard meterological data. Calls get_stand_meteo.
        It also checks to make sure that the years that were requested are
        available. Data is not available for the same years at all the buoys.

        Returns
        -------
        df : pandas dataframe
            Contains all the data from all the years that were specified
            in year_range.
        """

        start, stop = year_range
        df = pd.DataFrame()  # initialize empty df
        for ii in range(start, stop + 1):
            data = self.get_stand_meteo(buoy, ii)
            new_df = data
            df = df.append(new_df)
        return df


class Formatter:
    """
    Correctly formats the data contained in the link into a
    pandas dataframe.
    """

    def __init__(self, link):
        self.link = link

    def format_stand_meteo(self):
        """
        Format the standard Meteorological data.
        """

        df = pd.read_csv(self.link, delim_whitespace=True,
                         na_values=[99, 999, 9999, 99.00, 999.0, 9999.0])

        # 2007 and on format
        if df.iloc[0, 0] == '#yr':

            df = df.rename(columns={'#YY': 'YY'})  # get rid of hash

            # make the indices
            date_str = df.YY + ' ' + df.MM + ' ' + df.DD + ' ' + df.hh + ' ' + df.mm
            df.drop(0, inplace=True)  # first row is units, so drop them
            ind = pd.to_datetime(date_str.drop(0), format="%Y %m %d %H %M")

            df.index = ind

            # drop useless columns and rename the ones we want
            df.drop(['YY', 'MM', 'DD', 'hh', 'mm'], axis=1, inplace=True)
            df.columns = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES',
                          'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']


        # before 2006 to 2000
        else:
            date_str = df.YYYY.astype('str') + ' ' + df.MM.astype('str') + \
                       ' ' + df.DD.astype('str') + ' ' + df.hh.astype('str')

            ind = pd.to_datetime(date_str, format="%Y %m %d %H")

            df.index = ind

            # drop useless columns and rename the ones we want
            #######################
            '''FIX MEEEEE!!!!!!!
            Get rid of the try except
            some have minute column'''

            # this is hacky and bad
            try:
                df.drop(['YYYY', 'MM', 'DD', 'hh', 'mm'], axis=1, inplace=True)
                df.columns = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES',
                              'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']

            except:
                df.drop(['YYYY', 'MM', 'DD', 'hh'], axis=1, inplace=True)
                df.columns = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES',
                              'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']

        # all data should be floats
        df = df.astype('float')
        nvals = [99, 999, 9999, 99.0, 999.0, 9999.0]
        df.replace(nvals, np.nan, inplace=True)

        return df


class GetMonths(Formatter):
    """
    Before a year is complete ndbc stores there data monthly.
    This class will get all that scrap data.
    """

    def __init__(self, buoy, year=None):
        self.buoy = buoy
        self.year = year

    def get_stand_meteo(self):
        # see what is on the NDBC so we only pull the years that are available
        links = []

        # need to also retrieve jan, feb, march, etc.
        month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c']  # for the links

        # NDBC sometimes lags the new months in january and feb
        # Might need to define a year on init
        if not self.year:
            self.year = str(datetime.date.today().year)

            if datetime.date.month <= 2:
                print
                "using" + self.year + "to get the months. Might be wrong!"

        # for contstructing links
        base = 'http://www.ndbc.noaa.gov/view_text_file.php?filename='
        base2 = 'http://www.ndbc.noaa.gov/data/stdmet/'
        mid = '.txt.gz&dir=data/stdmet/'

        for ii in range(len(month)):

            # links can come in 2 formats
            link = base + str(self.buoy) + str(k[ii]) + self.year + mid + str(month[ii]) + '/'
            link2 = base2 + month[ii] + '/' + str(self.buoy) + '.txt'

            try:
                urllib.urlopen(link)
                links.append(link)

            except:
                print(str(month[ii]) + '2015' + ' not in records')
                print
                link

            # need to try the second link
            try:
                urllib.urlopen(link2)
                links.append(link2)
                print(link2 + 'was found in records')
            except:
                pass

        # start grabbing some data
        df = pd.DataFrame()

        for L in links:
            self.link = L
            new_df = self.format_stand_meteo()
            print
            'Link : ' + L
            df = df.append(new_df)

        return df


class GetHistoric(Formatter):

    def __init__(self, buoy, year, year_range=None):
        self.buoy = buoy
        self.year = year

    def hist_stand_meteo(self, link=None):
        '''
        Standard Meteorological Data. Data header was changed in 2007. Thus
        the need for the if statement below.



        WDIR	Wind direction (degrees clockwise from true N)
        WSPD	Wind speed (m/s) averaged over an eight-minute period
        GST		Peak 5 or 8 second gust speed (m/s)
        WVHT	Significant wave height (meters) is calculated as
                the average of the highest one-third of all of the
                wave heights during the 20-minute sampling period.
        DPD		Dominant wave period (seconds) is the period with the maximum wave energy.
        APD		Average wave period (seconds) of all waves during the 20-minute period.
        MWD		The direction from which the waves at the dominant period (DPD) are coming.
                (degrees clockwise from true N)
        PRES	Sea level pressure (hPa).
        ATMP	Air temperature (Celsius).
        WTMP	Sea surface temperature (Celsius).
        DEWP	Dewpoint temperature
        VIS		Station visibility (nautical miles).
        PTDY	Pressure Tendency
        TIDE	The water level in feet above or below Mean Lower Low Water (MLLW).
        '''

        if not link:
            base = 'http://www.ndbc.noaa.gov/view_text_file.php?filename='
            link = base + str(self.buoy) + 'h' + str(self.year) + '.txt.gz&dir=data/historical/stdmet/'

        # combine the first five date columns YY MM DD hh and make index
        df = pd.read_csv(link, delim_whitespace=True, na_values=[99, 999, 9999, 99.0, 999.0, 9999.0])

        # 2007 and on format
        if df.iloc[0, 0] == '#yr':

            df = df.rename(columns={'#YY': 'YY'})  # get rid of hash

            # make the indices
            date_str = df.YY + ' ' + df.MM + ' ' + df.DD + ' ' + df.hh + ' ' + df.mm
            df.drop(0, inplace=True)  # first row is units, so drop them
            ind = pd.to_datetime(date_str.drop(0), format="%Y %m %d %H %M")

            df.index = ind

            # drop useless columns and rename the ones we want
            df.drop(['YY', 'MM', 'DD', 'hh', 'mm'], axis=1, inplace=True)
            df.columns = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES',
                          'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']


        # before 2006 to 2000
        else:
            date_str = df.YYYY.astype('str') + ' ' + df.MM.astype('str') + \
                       ' ' + df.DD.astype('str') + ' ' + df.hh.astype('str')

            ind = pd.to_datetime(date_str, format="%Y %m %d %H")

            df.index = ind

            # drop useless columns and rename the ones we want
            #######################
            '''FIX MEEEEE!!!!!!!
            Get rid of the try except
            some have minute column'''

            # this is hacky and bad
            try:
                df.drop(['YYYY', 'MM', 'DD', 'hh', 'mm'], axis=1, inplace=True)
                df.columns = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES',
                              'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']

            except:
                df.drop(['YYYY', 'MM', 'DD', 'hh'], axis=1, inplace=True)
                df.columns = ['WDIR', 'WSPD', 'GST', 'WVHT', 'DPD', 'APD', 'MWD', 'PRES',
                              'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE']

        # all data should be floats
        df = df.astype('float')
        nvals = [99, 999, 9999, 99.0, 999.0, 9999.0]
        df.replace(nvals, np.nan, inplace=True)

        return df


class Makecall(GetHistoric, GetMonths):

    def __init__(self, year_range):
        self.year_range = year_range

    def get_all_stand_meteo(self):
        """
        Retrieves all the standard meterological data. Calls get_stand_meteo.
        It also checks to make sure that the years that were requested are
        available. Data is not available for the same years at all the buoys.

        Returns
        -------
        df : pandas dataframe
            Contains all the data from all the years that were specified
            in year_range.
        """

        start_yr, stop_yr = self.year_range

        # see what is on the NDBC so we only pull the years that are available
        links = []
        for ii in range(start_yr, stop_yr + 1):

            base = 'http://www.ndbc.noaa.gov/view_text_file.php?filename='
            end = '.txt.gz&dir=data/historical/stdmet/'
            link = base + str(self.buoy) + 'h' + str(ii) + end

            try:
                urllib.request.urlopen(link)
                links.append(link)

            except:
                print(str(ii) + ' not in records')

        # need to also retrieve jan, feb, march, etc.
        month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c']  # for the links

        for ii in range(len(month)):
            mid = '.txt.gz&dir=data/stdmet/'
            link = base + str(self.buoy) + str(k[ii]) + '2015' + mid + str(month[ii]) + '/'

            try:
                urllib.request.urlopen(link)
                links.append(link)

            except:
                print(str(month[ii]) + '2015' + ' not in records')
                print(link)

        # start grabbing some data
        df = pd.DataFrame()  # initialize empty df

        for L in links:
            new_df = self.get_stand_meteo(link=L)
            print('Link : ' + L)
            df = df.append(new_df)

        return df
