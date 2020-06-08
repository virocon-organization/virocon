
import matplotlib.pyplot as plt
import seaborn as sns
import viroconcom.dataNDBC as ndbc

# Example for one year.
buoy = 41108
date = "2017-02-11/to/2018-11-27"

H = ndbc.HistoricData()
df = H.get_data(buoy, date)

# Plotting three subplots.
# Plot significant wave height.
sub1 = plt.subplot(2, 2, 1)
df.WVHT.plot()
sub1.set_ylabel('Significant wave height (m)', fontsize=14)
sub1.set_xlabel('Date')

# Plot average wave period.
sub2 = plt.subplot(2, 2, 3)
df.APD.plot()
sub2.set_ylabel('Average wave period (s)', fontsize=14)
sub2.set_xlabel('Date')

# Scatterplot of the data.
sub3 = plt.subplot(2, 2, (2, 4))
plt.scatter(df.WVHT, df.APD)
sub3.set_xlabel('Significant wave height (m)', fontsize=14)
sub3.set_ylabel('Average wave period (s)', fontsize=14)
sns.despine()
plt.show()