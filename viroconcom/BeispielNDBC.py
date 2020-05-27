import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import viroconcom.buoypy as bp
buoy = 46071
year = 2006

H = bp.Historic_Data()
df = H.get_stand_meteo(buoy,year)

# plotting
fig,ax = plt.subplots(2,sharex=True)
df.WVHT.plot(ax=ax[0])
ax[0].set_ylabel('Wave Height (m)',fontsize=14)

df.APD.plot(ax=ax[1])
ax[1].set_ylabel('Average Wave Periode (s)',fontsize=14)
ax[1].set_xlabel('')
sns.despine()
plt.show()



buoy = 46071
year_range = (2004,2005)

H = bp.Historic_Data()
X = H.get_all_stand_meteo(buoy, year_range)
print(X)
#plotting
fig,ax = plt.subplots(2,sharex=True)
X.WVHT.plot(ax=ax[0])
ax[0].set_ylabel('Wave Height (m)',fontsize=14)
X.APD.plot(ax=ax[1])
ax[1].set_ylabel('Average Wave Periode (s)',fontsize=14)
ax[1].set_xlabel('')
sns.despine()
plt.show()