
import matplotlib.pyplot as plt
import seaborn as sns
import viroconcom.dataNDBC as ndbc

# Example for one year.
buoy = 46071
year = 2004

H = ndbc.HistoricData()
df = H.get_stand_meteo(buoy, year)

# Plotting.
fig, ax = plt.subplots(2)
df.WVHT.plot(ax=ax[0])
ax[0].set_ylabel('Wave Height (m)', fontsize=14)

df.APD.plot(ax=ax[1])
ax[1].set_ylabel('Average Wave Period (s)',fontsize=14)
ax[1].set_xlabel('')
sns.despine()
plt.show()

# Example for multiple years

buoy = 41108
year_range = (2013, 2018)

H = ndbc.HistoricData()
X = H.get_all_stand_meteo(buoy, year_range)

# Plotting.
fig, ax = plt.subplots(2)
X.WVHT.plot(ax=ax[0])
ax[0].set_ylabel('Wave Height (m)', fontsize=14)
X.APD.plot(ax=ax[1])
ax[1].set_ylabel('Average Wave Period (s)', fontsize=14)
ax[1].set_xlabel('')
sns.despine()
plt.show()