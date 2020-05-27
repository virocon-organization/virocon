"""

Functions to query the NDBC (http://www.ndbc.noaa.gov/).

Info about all of noaa data can be found at:
http://www.ndbc.noaa.gov/docs/ndbc_web_data_guide.pdf

What all the values mean:
http://www.ndbc.noaa.gov/measdes.shtml

"""

import datetime
import pandas as pd
from sqlalchemy import create_engine


class Historic_Data:

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

        #combine the first five date columns YY MM DD hh and make index
        df = pd.read_csv(link, header=0, delim_whitespace=True, dtype=object,
                         na_values=[99, 999, 9999, 99., 999., 9999., 99.00])

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

        links = []
        df = pd.DataFrame()  # initialize empty df
        for ii in range(start, stop + 1):
            data = self.get_stand_meteo(buoy, ii)
            new_df = data
            df = df.append(new_df)
        return df


class write_data(Historic_Data):

    def __init__(self, buoy, year, year_range,db_name = 'buoydata.db'):
        self.buoy = buoy
        self.year = year
        self.year_range=year_range
        self.db_name = db_name

    def write_all_stand_meteo(self):
        """
        Write the standard meteological data to the database. See get_all_stand_meteo
        for a discription of the data. Which is in the historic data class.

        Returns
        -------
        df : pandas dataframe (date, frequency)
            data frame containing the raw spectral data. index is the date
            and the columns are each of the frequencies

        """

        #hist = self.historic_data(self.buoy,self.year,year_range=self.year_range)
        df = self.get_all_stand_meteo()

        #write the df to disk
        disk_engine = create_engine('sqlite:///' + self.db_name)

        table_name = str(self.buoy) + '_buoy'
        df.to_sql(table_name,disk_engine,if_exists='append')
        sql = disk_engine.execute("""DELETE FROM wave_data
            WHERE rowid not in
            (SELECT max(rowid) FROM wave_data GROUP BY date)""")

        print(str(self.buoy) + 'written to database : ' + str(self.db_name))


        return True


class read_data:
    """
    Reads the data from the setup database
    """

    def __init__(self, buoy, year_range=None):
        self.buoy = buoy
        self.year_range = year_range
        self.disk_eng = 'sqlite:///buoydata.db'


    def get_stand_meteo(self):

        disk_engine = create_engine(self.disk_eng)


        df = pd.read_sql_query(" SELECT * FROM " + "'" + str(self.buoy) + '_buoy' + "'", disk_engine)

        #give it a datetime index since it was stripped by sqllite
        df.index = pd.to_datetime(df['index'])
        df.index.name='date'
        df.drop('index',axis=1,inplace=True)

        if self.year_range:
            print("""this is not implemented in SQL. Could be slow.
                    Get out while you can!!!""" )

            start,stop = (self.year_range)
            begin = df.index.searchsorted(datetime.datetime(start, 1, 1))
            end = df.index.searchsorted(datetime.datetime(stop, 12, 31))
            df = df.ix[begin:end]



        return df
