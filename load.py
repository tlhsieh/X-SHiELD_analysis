import numpy as np
import pandas as pd
import xarray as xr

import os
import datetime
import dateutil

from util import date_linspace

import sys
sys.path.append('/home/tlh/ipy')
from gfd import xrinterp

def load_land():
    land_frac = xr.open_dataset('/archive/Liwei.Jia/spear_med/rf_hist/fcst/s_j11_OTA_IceAtmRes_L33/i20191101_OTA_IceAtmRes_L33_update/pp_ens_01/land/land.static.nc')['land_frac']

    return land_frac

def load_zsurf():
    zsurf = xr.open_dataset('/archive/kyc/Stellar/20191020.00Z.C3072.L79x2_pire/history/2020010800/zsurf_coarse_C3072_1440x720.fre.nc')['zsurf_coarse']

    return zsurf

def load_hgt():
    hgt = xr.open_dataset('/archive/kyc/Stellar/20191020.00Z.C3072.L79x2_pire/history/2019102000/HGTsfc_C3072_11520x5760.fre.nc')['HGTsfc']

    return hgt

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
    
def load_shield_pp(monthlist, yr, field='pr', exp='', coarse_grain=True):
    """load post-processed monthly mean data"""

    mo_dim = pd.Index(monthlist, name='time')
    
    if coarse_grain:
        target_grid = xr.open_dataarray(f'/archive/tlh/pp_xshield/20191020.00Z.C3072.L79x2_pire/monthly/202001.tsfc_coarse.nc')
        da = xr.concat([xrinterp(xr.open_dataarray(f'/archive/tlh/pp_xshield/20191020.00Z.C3072.L79x2_pire{exp}/monthly/{yr}{mo:02d}.{field}.nc'), target_grid) for mo in monthlist], dim=mo_dim)
    else:
        da = xr.concat([xr.open_dataarray(f'/archive/tlh/pp_xshield/20191020.00Z.C3072.L79x2_pire{exp}/monthly/{yr}{mo:02d}.{field}.nc') for mo in monthlist], dim=mo_dim)

    return da

def _spear_convention(date):
    """see /archive/Liwei.Jia/spear_med/rf_hist/fcst/s_j11_OTA_IceAtmRes_L33/README"""
    
    year = date[:4]
    if int(year) <= 2019:
        return f'i{date}01_OTA_IceAtmRes_L33_update/'
    elif year == '2020':
        return f'i{date}01_OTA_IceAtmRes_L33_rerun/'
    elif year == '2021':
        return f'i{date}01_OTA_IceAtmRes_L33_update/'
    else:
        return f'i{date}01_OTA_IceAtmRes_L33/'

def _plus11mo(date):
    """
    in (str)
    out (str)
    """
    
    date_out = datetime.datetime.strptime(date, '%Y%m') + dateutil.relativedelta.relativedelta(months=11)
    
    return date_out.strftime('%Y%m')

def _plus11mo_ymd(date):
    """
    in (str)
    out (str)
    """
    
    date_out = datetime.datetime.strptime(date, '%Y%m') + dateutil.relativedelta.relativedelta(months=12) - dateutil.relativedelta.relativedelta(days=1)
    
    return date_out.strftime('%Y%m%d')

def _ndays_in_month(date, lead=0):
    """
    in (str)
    out (int)
    
    Number of days in the (current + lead) month
    """
    
    datetime_curr = datetime.datetime.strptime(date, '%Y%m')
    datetime_beg = datetime_curr + dateutil.relativedelta.relativedelta(months=lead)
    datetime_end = datetime_curr + dateutil.relativedelta.relativedelta(months=lead+1)
    
    return int((datetime_end - datetime_beg)/datetime.timedelta(days=1))

def load_spear_monthly(datebeg, dateend, lead=1, ens='01', field='precip'):
    ## collect a list of months in which simulation is initialized
    monthbeg = datetime.datetime.strptime(datebeg, '%Y%m') - dateutil.relativedelta.relativedelta(months=lead)
    monthend = datetime.datetime.strptime(dateend, '%Y%m') - dateutil.relativedelta.relativedelta(months=lead)

    months = [monthbeg]
    monthcurr = monthbeg + dateutil.relativedelta.relativedelta(months=1)
    while monthcurr <= monthend:
        months.append(monthcurr)
        monthcurr = monthcurr + dateutil.relativedelta.relativedelta(months=1)

    filemonths = [month.strftime('%Y%m') for month in months]
    
    ## collect file paths
    filenames = [f'/archive/Liwei.Jia/spear_med/rf_hist/fcst/s_j11_OTA_IceAtmRes_L33/{_spear_convention(month)}pp_ens_{ens}/atmos/ts/monthly/1yr/atmos.{month}-{_plus11mo(month)}.{field}.nc' for month in filemonths]
    print(filenames[0])
    
    ## dmget all files
    os.system('dmget '+' '.join(filenames))
    
    da = xr.concat([xr.open_dataset(filenames[i], chunks={'time': 1})[field].isel(time=1) for i in range(len(filenames))], dim='time')
    # print(da.time.values)

    ## convert units
    da = _convert_units(da, field)
    
    return da

def load_spear_daily(datebeg, dateend, lead=1, ens='01', field='precip'):
    ## collect a list of months in which simulation is initialized
    monthbeg = datetime.datetime.strptime(datebeg[:6], '%Y%m') - dateutil.relativedelta.relativedelta(months=lead)
    monthend = datetime.datetime.strptime(dateend[:6], '%Y%m') - dateutil.relativedelta.relativedelta(months=lead)

    months = [monthbeg]
    monthcurr = monthbeg + dateutil.relativedelta.relativedelta(months=1)
    while monthcurr <= monthend:
        months.append(monthcurr)
        monthcurr = monthcurr + dateutil.relativedelta.relativedelta(months=1)

    filemonths = [month.strftime('%Y%m') for month in months]
    
    ## collect file paths
    filenames = [f'/archive/Liwei.Jia/spear_med/rf_hist/fcst/s_j11_OTA_IceAtmRes_L33/{_spear_convention(month)}pp_ens_{ens}/atmos_daily/ts/daily/1yr/atmos_daily.{month}01-{_plus11mo_ymd(month)}.{field}.nc' for month in filemonths]
    print(filenames[0])
    
    ## dmget all files
    os.system('dmget '+' '.join(filenames))
    
    ndays_month0 = [_ndays_in_month(month, lead=0) for month in filemonths]
    ndays_month1 = [_ndays_in_month(month, lead=1) for month in filemonths]
    
    da = xr.concat([xr.open_dataset(filenames[i], chunks={'time': 1})[field].isel(time=range(ndays_month0[i], (ndays_month0[i]+ndays_month1[i]))) for i in range(len(filenames))], dim='time')
    # print(da.time.values)
    
    ## include end point in date range
    datebeg = datetime.datetime.strptime(datebeg, '%Y%m%d')
    datebeg = datebeg.strftime('%Y-%m-%d')
    dateend = datetime.datetime.strptime(dateend, '%Y%m%d')
    dateend = dateend + datetime.timedelta(days=1)
    dateend = dateend.strftime('%Y-%m-%d')
    
    ## select dates
    # itrange = range(*np.searchsorted(da.time.astype('datetime64[ns]'), [np.datetime64(datebeg), np.datetime64(dateend)]))
    ## select dates
    itbeg, itend = np.searchsorted(da.time.astype('datetime64[ns]'), [np.datetime64(datebeg), np.datetime64(dateend)])
    itrange = range(itbeg, itend) # no need for itend+1 for time mean data
    da = da.isel(time=itrange)
    # print(da.time.values)

    ## convert units
    da = _convert_units(da, field)
    
    return da

def load_spear_4xdaily(datebeg, dateend, lead=1, ens='01', field='precip'):
    ## collect a list of months in which simulation is initialized
    monthbeg = datetime.datetime.strptime(datebeg[:6], '%Y%m') - dateutil.relativedelta.relativedelta(months=lead)
    monthend = datetime.datetime.strptime(dateend[:6], '%Y%m') - dateutil.relativedelta.relativedelta(months=lead)

    months = [monthbeg]
    monthcurr = monthbeg + dateutil.relativedelta.relativedelta(months=1)
    while monthcurr <= monthend:
        months.append(monthcurr)
        monthcurr = monthcurr + dateutil.relativedelta.relativedelta(months=1)

    filemonths = [month.strftime('%Y%m') for month in months]
    
    ## collect file paths
    filenames = [f'/archive/Liwei.Jia/spear_med/rf_hist/fcst/s_j11_OTA_IceAtmRes_L33/{_spear_convention(month)}pp_ens_{ens}/atmos_4xdaily_avg/ts/6hr/1yr/atmos_4xdaily_avg.{month}0100-{_plus11mo_ymd(month)}23.{field}.nc' for month in filemonths]
    print(filenames[0])
    
    ## dmget all files
    os.system('dmget '+' '.join(filenames))
    
    ndays_month0 = [_ndays_in_month(month, lead=0) for month in filemonths]
    ndays_month1 = [_ndays_in_month(month, lead=1) for month in filemonths]
    
    da = xr.concat([xr.open_dataset(filenames[i], chunks={'time': 1})[field].isel(time=range(ndays_month0[i]*4, (ndays_month0[i]+ndays_month1[i])*4)) for i in range(len(filenames))], dim='time')
    # print(da.time.values)
    
    ## include end point in date range
    datebeg = datetime.datetime.strptime(datebeg, '%Y%m%d')
    datebeg = datebeg.strftime('%Y-%m-%d')
    dateend = datetime.datetime.strptime(dateend, '%Y%m%d')
    dateend = dateend + datetime.timedelta(days=1)
    dateend = dateend.strftime('%Y-%m-%d')
    
    ## select dates
    # itrange = range(*np.searchsorted(da.time.astype('datetime64[ns]'), [np.datetime64(datebeg), np.datetime64(dateend)]))
    ## select dates
    itbeg, itend = np.searchsorted(da.time.astype('datetime64[ns]'), [np.datetime64(datebeg), np.datetime64(dateend)])
    itrange = range(itbeg, itend+1) # itend+1 to include hour 00 in the following day for consistency
    da = da.isel(time=itrange)
    print(da.time.values)

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
    if field in ['snow']:
        module = 'land'
    else:
        module = 'atmos'

    ## collect file paths
    filepaths = [f'/archive/Ming.Zhao/awg/warsaw_201710/c192L33_am4p0_2010climo_new{exp}/gfdl.ncrc4-intel-prod-openmp/pp/{module}/ts/monthly/1yr/{module}.{yr:04d}01-{yr:04d}12.{field}.nc' for yr in range(yrbeg, yrend+1)]
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