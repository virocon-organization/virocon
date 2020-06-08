
import matplotlib.pyplot as plt
import seaborn as sns
import viroconcom.dataNDBC as ndbc

# Example for one year.
buoy = 41108
date = "2017-02-11/to/2018-11-27"

H = ndbc.HistoricData()
df = H.get_data(buoy, date)

# Plotting three subplots.
fig, ax = plt.subplots(3)

# Plot significant wave height.
df.WVHT.plot(ax=ax[0])
ax[0].set_ylabel('Significant wave height (m)', fontsize=14)

# Plot average wave period.
df.APD.plot(ax=ax[1])
ax[1].set_ylabel('Average wave period (s)', fontsize=14)
ax[1].set_xlabel('')

# Scatterplot of the data.
plt.scatter(df.WVHT, df.APD)
ax[2].set_xlabel('Significant wave height (m)')
ax[2].set_ylabel('Average wave period (s)')
sns.despine()
plt.show()