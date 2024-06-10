import numpy as np
import xarray as xr

from load import load_shield, load_am4_monthly
from param import shield_dates, shield_months
from util import datenum2txt, first_and_last_days_of_month

def shield():
    datebeg, dateend = shield_dates()[0]
    # datebeg, dateend = '20200101', '20200131'
    # datebeg, dateend = '20210101', '20210131'
    # datebeg, dateend = '20220101', '20220112'

    da = load_shield(datebeg, dateend, 'tsfc_coarse', exp=exp, coarse=True).mean('time', keep_attrs=True)

    output_name = f'ar_data/t_surf/tsfc_coarse_mean_X-SHiELD{exp}_{datenum2txt(datebeg)}-{datenum2txt(dateend)}.nc'

    return da, output_name

def am4():
    yrbeg, yrend = 11, 100 #i

    da_monthly = load_am4_monthly([1, 11, 12], yrbeg=yrbeg, yrend=yrend, field='precip', exp=exp)
    # da_monthly = load_am4_monthly([1], yrbeg=yrbeg, yrend=yrend, field='t_surf', exp=exp)

    da = da_monthly.groupby('time.year').mean('time', keep_attrs=True) # Note: month 1, 11, 12 in the same year are used to compute the mean

    output_name = f'ar_data/precip_AM4{exp}_yr{yrbeg:04d}-{yrend:04d}_JanNovDec.nc'
    # output_name = f'ar_data/t_surf/t_surf_mean_AM4{exp}_yr{yrbeg:04d}-{yrend:04d}_Jan.nc'

    return da, output_name

def shield_monthly_mean(month='201911', field='pr', exp='', coarse=False):
    datebeg, dateend = first_and_last_days_of_month(month)

    da = load_shield(datebeg, dateend, field, exp=exp, coarse=coarse).mean('time', keep_attrs=True)

    output_name = f'/archive/tlh/pp_xshield/20191020.00Z.C3072.L79x2_pire{exp}/monthly/{month}.{field}.nc'

    return da, output_name

if __name__ == "__main__":

    # for exp in ['_PLUS_4K_CO2_1270ppmv']:
    #     for field in ['t850_coarse', 't500_coarse']:
    #         for month in shield_months():
    #             da, output_name = shield_monthly_mean(month, field, exp, coarse=True)
    #             da.to_dataset().to_netcdf(output_name)
            
    #####
    
    exp = ''
    # exp = '_PLUS_4K'
    # exp = '_CO2_1270ppmv'
    # exp = '_PLUS_4K_CO2_1270ppmv'

    # for field in ['ULWRFtoa_coarse', 'USWRFsfc_coarse', 'DSWRFsfc_coarse', 'omg500_coarse', 'u850_coarse', 'v850_coarse', 'tsfc_coarse']:
    # for field in ['LHTFLsfc_coarse', 'USWRFtoa_coarse', 'ULWRFsfc_coarse', 'DLWRFsfc_coarse', 'DSWRFtoa_coarse']:
    #     for month in shield_months():
    #         da, output_name = shield_monthly_mean(month, field, exp, coarse=True)
    #         da.to_dataset().to_netcdf(output_name)

    for field in ['pr', 'snowd', 'uas', 'vas', 'tas']:
        for month in shield_months():
            da, output_name = shield_monthly_mean(month, field, exp, coarse=False)
            da.to_dataset().to_netcdf(output_name)

    #####

    # for exp in ['', '_PLUS_4K']:
    #     da, output_name = shield()
    #     da.to_dataset().to_netcdf(output_name)

    # for exp in ['', '_p4K']:
    #     da, output_name = am4()
    #     da.to_dataset().to_netcdf(output_name)
