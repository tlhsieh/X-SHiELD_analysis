import numpy as np
import xarray as xr

import os
import datetime

from util import date_linspace

def load_land():
    land_bool_spear = xr.open_dataset('/archive/Liwei.Jia/spear_med/rf_hist/fcst/s_j11_OTA_IceAtmRes_L33/i20191101_OTA_IceAtmRes_L33_update/pp_ens_01/land/land.static.nc')['land_frac'] > 0

    return land_bool_spear

def load_tas():
    tas = xr.open_dataset('/archive/kyc/Stellar/20191020.00Z.C3072.L79x2_pire/history/2020010800/tas_C3072_11520x5760.fre.nc').tas[0]

    return tas

def _convert_units(da, field):
    if (field == 'pr') or (field == 'precip'):
        new_name = ' '.join(da.long_name.split(' ')[:-1]+['[mm/day]'])
        print(f"Note: long_name changed from {da.attrs['long_name']} to {new_name}")
        da = da*86400 # attribute 'long_name' is lost after operation
        da.attrs['long_name'] = new_name

    return da

def _folder_list(exp):
    """all available files, see /archive/kyc"""
    
    if exp == '':
        return \
    [f'/archive/kyc/Stellar/20191020.00Z.C3072.L79x2_pire/history/{date}00/' for date in date_linspace('20191020', '20201223', delta_day=5)] + \
    [f'/archive/kyc/Stellar_new/20191020.00Z.C3072.L79x2_pire/history/{date}00/' for date in date_linspace('20201228', '20210522', delta_day=5)] + \
    [f'/archive/kyc/Stellar_new/20191020.00Z.C3072.L79x2_pire/history/{date}00/' for date in date_linspace('20210527', '20220112', delta_day=1)]
    elif exp == '_PLUS_4K_CO2_1270ppmv':
        return \
    [f'/archive/kyc/Stellar/20191020.00Z.C3072.L79x2_pire{exp}/history/{date}00/' for date in date_linspace('20191020', '20200507', delta_day=5)] + \
    [f'/archive/kyc/Stellar/20191020.00Z.C3072.L79x2_pire{exp}/history/{date}00/' for date in date_linspace('20200512', '20201227', delta_day=1)] + \
    [f'/archive/kyc/Stellar_new/20191020.00Z.C3072.L79x2_pire{exp}/history/{date}00/' for date in date_linspace('20201228', '20210109', delta_day=2)]
    else:
        return \
    [f'/archive/kyc/Stellar/20191020.00Z.C3072.L79x2_pire{exp}/history/{date}00/' for date in date_linspace('20191020', '20201223', delta_day=5)] + \
    [f'/archive/kyc/Stellar_new/20191020.00Z.C3072.L79x2_pire{exp}/history/{date}00/' for date in date_linspace('20201228', '20220128', delta_day=2)]

def load_shield(datebeg, dateend, field, exp='', coarse=True):
    if coarse == True:
        res = '1440x720'
    else:
        res = '11520x5760'
        
    ## collect file paths
    all_dates = [np.datetime64(folder[-11:-3]) for folder in _folder_list(exp)]
    ifilebeg = np.searchsorted(all_dates, np.datetime64(datebeg), side='right') - 1 # -1 to include the first file
    ifileend = np.searchsorted(all_dates, np.datetime64(dateend), side='right') # side='right' to include end point

    filepaths = [f'{folder}{field}_C3072_{res}.fre.nc' for folder in _folder_list(exp)[ifilebeg:ifileend]]
    print(filepaths[0])
    
    ## dmget all files
    os.system('dmget '+' '.join(filepaths))

    da = xr.concat([xr.open_dataset(filepath, chunks={'time': 1})[field] for filepath in filepaths], dim='time')
    
    ## include end point in date range
    datebeg = datetime.datetime.strptime(datebeg, '%Y%m%d')
    datebeg = datebeg.strftime('%Y-%m-%d')
    dateend = datetime.datetime.strptime(dateend, '%Y%m%d')
    dateend = dateend + datetime.timedelta(days=1)
    dateend = dateend.strftime('%Y-%m-%d')
    
    ## select dates
    itbeg, itend = np.searchsorted(da.time.astype('datetime64[ns]'), [np.datetime64(datebeg), np.datetime64(dateend)])
    itrange = range(itbeg, itend+1) # itend+1 to include hour 00 in the following day for consistency
    da = da.isel(time=itrange)

    ## convert units
    da = _convert_units(da, field)
    
    return da
    
def load_am4_8xdaily(monthlist=range(1, 12+1), yrbeg=11, yrend=20, field='pr', exp=''):
    ## collect file paths
    filepaths = [f'/archive/Ming.Zhao/awg/warsaw_201710/c192L33_am4p0_2010climo_new{exp}/gfdl.ncrc4-intel-prod-openmp/pp/atmos_cmip/ts/3hr/1yr/atmos_cmip.{yr:04d}010100-{yr:04d}123123.{field}.nc' for yr in range(yrbeg, yrend+1)]
    print(filepaths[0])

    ## dmget all files
    os.system('dmget '+' '.join(filepaths))

    imonthlist = [i - 1 for i in monthlist]

    da = xr.concat([list(xr.open_dataset(filepath)[field].groupby('time.month'))[imo][-1] for filepath in filepaths for imo in imonthlist], dim='time') # [-1] to extract data from the list([month, data]) returned by groupby('time.month')

    ## convert units
    da = _convert_units(da, field)

    return da

def load_am4_monthly(monthlist=range(1, 12+1), yrbeg=11, yrend=20, field='precip', exp=''):
    ## collect file paths
    filepaths = [f'/archive/Ming.Zhao/awg/warsaw_201710/c192L33_am4p0_2010climo_new{exp}/gfdl.ncrc4-intel-prod-openmp/pp/atmos/ts/monthly/1yr/atmos.{yr:04d}01-{yr:04d}12.{field}.nc' for yr in range(yrbeg, yrend+1)]
    print(filepaths[0])

    ## dmget all files
    os.system('dmget '+' '.join(filepaths))

    imonthlist = [i - 1 for i in monthlist]

    da = xr.concat([xr.open_dataset(filepath)[field].isel(time=imo) for filepath in filepaths for imo in imonthlist], dim='time')

    ## convert units
    da = _convert_units(da, field)

    return da

if __name__ == "__main__":
    # da = load_am4_8xdaily([1, 2, 12], yrend=12)
    # print(da.time.values) # check if all months show up

    da = load_shield('20191020', '20191129', 'pr', exp='', coarse=False)
    print(da.time) # check if first and last points are correct