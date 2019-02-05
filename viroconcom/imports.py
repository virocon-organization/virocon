import matplotlib.pyplot as plt
import pandas as pd


class NDBCHistoricData:

    def __init__(self, buoy, year):

        link = 'http://www.ndbc.noaa.gov/view_text_file.php?filename='
        link += '{}h{}.txt.gz&dir=data/historical/'.format(buoy, year)
        self.link = link
        # self.year_range = year_range
        # self.buoy = buoy

    def get_stand_meteo(self, link=None):
        """
        Standard Meteorological Data. Data header was changed in 2007. Thus
        the need for the if statement below.


        WSPD    Wind speed (m/s) averaged over an eight-minute period
        WVHT    Significant wave height (meters) is calculated as
                the average of the highest one-third of all of the
                wave heights during the 20-minute sampling period.
        APD     Average wave period (seconds) of all waves during the 20-minute period.
        """

        link = self.link + 'stdmet/'

        # combine the first five date columns YY MM DD hh and make index
        df = pd.read_csv(link, header=0, delim_whitespace=True, dtype=object,
                         na_values=[99, 999, 9999, 99., 999., 9999.])

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
            df.drop(['WDIR', 'GST', 'DPD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE'], axis=1, inplace=True)
            df.columns = ['WSPD', 'WVHT', 'APD']


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

            df.drop(['WDIR', 'GST', 'DPD', 'MWD', 'PRES', 'ATMP', 'WTMP', 'DEWP', 'VIS', 'TIDE'], axis=1, inplace=True)
            df.columns = ['WSPD', 'WVHT', 'APD']

        # all data should be floats
        df = df.astype('float')

        return df


class NDBCImport(NDBCHistoricData):
    def __init__(self, buoy, year=None, year_range=None):
        NDBCHistoricData.__init__(self, buoy, year)
        self.virocon_data = self.get_stand_meteo()
        self.year = year
        self.buoy = buoy

    def get_stand_meteo(self):
        return super().get_stand_meteo().dropna()  # or not?!

    # def write_to_csv(self):


if __name__ == "__main__":
    myNDBC = NDBCImport(44020, year=2015)
    print(myNDBC.virocon_data)
    myPath = ''
    myPath += '{}-{}.csv'.format(myNDBC.buoy, myNDBC.year)
    myNDBC.virocon_data.to_csv(path_or_buf=myPath, index=False)
    fig, ax = plt.subplots(3, sharex=True)
    myNDBC.virocon_data.WSPD.plot(ax=ax[0])
    ax[0].set_ylabel('Wind speed (m/s)', fontsize=14)

    myNDBC.virocon_data.WVHT.plot(ax=ax[1])
    ax[1].set_ylabel('Wave hight (m)', fontsize=14)

    myNDBC.virocon_data.APD.plot(ax=ax[2])
    ax[2].set_ylabel('Average Period (sec)', fontsize=14)
    ax[2].set_xlabel('')
    fig.suptitle('station: {}, year: {}'.format(myNDBC.buoy, myNDBC.year), fontsize=16)
    plt.show()
