import numpy as np
import xarray as xr

from load import load_shield
from util import datenum2txt
from param import boundaries, shield_dates

import pandas as pd

xr.set_options(keep_attrs=True)

g = 9.81 # m/s2
L = 2.5e6 # J/kg

# exp = ''
# exp = '_PLUS_4K'
exp = '_CO2_1270ppmv'
# exp = '_PLUS_4K_CO2_1270ppmv'

for i in range(3):
    # (datebeg, dateend) = ('20191201', '20191215')
    (datebeg, dateend) = shield_dates()[i] #i

    ds_3d = xr.Dataset()

    fields_3d = ['u', 'v', 'q']
    levels = [1000, 925, 850, 700, 500, 200]
    pdim_Pa = pd.Index(np.array(levels)*100, name='p') # hPa to Pa

    for field in fields_3d:
        ds_3d[field] = xr.concat([load_shield(datebeg, dateend, field=f'{field}{level}_coarse', exp=exp, coarse=True) for level in levels], dim=pdim_Pa)
        ds_3d[field] = ds_3d[field].transpose('time', 'p', 'grid_yt_coarse', 'grid_xt_coarse')

    ps = load_shield('20191201', '20191231', field='ps_coarse', exp=exp, coarse=True).mean('time')

    criteria = xr.ones_like(ps)*(ds_3d.p)
    criteria = criteria < ps

    uflux = ds_3d['u']*ds_3d['q']
    vflux = ds_3d['v']*ds_3d['q']

    uflux = (uflux*uflux.p.diff('p')/(-g)*criteria).sum('p') # mass-weighted vertical integral
    vflux = (vflux*vflux.p.diff('p')/(-g)*criteria).sum('p') # mass-weighted vertical integral

    ds_mean = xr.Dataset()

    ds_mean['uflux'] = uflux.mean('time')
    ds_mean['vflux'] = vflux.mean('time')

    fields_2d = ['LHTFLsfc_coarse', 'PRATEsfc_coarse']

    for field in fields_2d:
        da = load_shield(datebeg, dateend, field=field, exp=exp, coarse=True).mean('time')
        if field == 'LHTFLsfc_coarse':
            ds_mean[field] = da/L # convert units to kg/m2/s
        else:
            ds_mean[field] = da # Note: precip's units are NOT converted to mm/day

    ds_mean.to_netcdf(f'ar_data/moisture_budget/moisture_budget_X-SHiELD{exp}_{datenum2txt(datebeg)}-{datenum2txt(dateend)}.nc')
