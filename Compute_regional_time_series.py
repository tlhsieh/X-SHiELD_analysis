## crop X-SHiELD time series for more efficient storage
import numpy as np
import xarray as xr
from load import load_shield
from util import crop, datenum2txt
from param import boundaries

# (datebeg, dateend) = ('20191201', '20191231')
(datebeg, dateend) = ('20191101', '20200131')
# (datebeg, dateend) = ('20201101', '20210131')
# (datebeg, dateend) = ('20200119', '20200313')

exp = ''
# exp = '_PLUS_4K'

# field = 'pr'
field = 'snowd'

da = load_shield(datebeg, dateend, field, exp, coarse=False)

step = 10 # load every 10 time steps at a time
indices = np.arange(0, len(da.time) - 1, step)
indices = np.append(indices, len(da.time))

domain = 'Western'
xlim, ylim = boundaries(domain)

da_crop = xr.concat([crop(da[indices[i]:indices[i+1]], xlim, ylim) for i in range(len(indices) - 1)], dim='time')

da_crop.to_dataset().to_netcdf(f'ar_data/regional/{field}_X-SHiELD{exp}_{datenum2txt(datebeg)}-{datenum2txt(dateend)}_{domain}.nc')