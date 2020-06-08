
import matplotlib.pyplot as plt
import seaborn as sns
import viroconcom.dataNDBC as ndbc

# Example for one year.
buoy = 41108
date = "2017-02-11/to/2018-11-27"

H = ndbc.HistoricData()
df = H.get_data(buoy, date)

# Plotting.
fig, ax = plt.subplots(2)
df.WVHT.plot(ax=ax[0])
ax[0].set_ylabel('Significant wave height (m)', fontsize=14)

df.APD.plot(ax=ax[1])
ax[1].set_ylabel('Average wave period (s)', fontsize=14)
ax[1].set_xlabel('')
sns.despine()
plt.show()