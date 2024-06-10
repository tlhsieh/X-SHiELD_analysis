import numpy as np
import xarray as xr

from load import load_shield
from param import shield_months
from util import first_and_last_days_of_month

def shield_monthly_mean_variance(month='201911', field1='vs_coarse', field2='vs_coarse', exp='', coarse=True):
    """compute second order statistics"""

    datebeg, dateend = first_and_last_days_of_month(month)

    da1 = load_shield(datebeg, dateend, field1, exp=exp, coarse=coarse)

    if field1 == field2:
        da2 = da1
    else:
        da2 = load_shield(datebeg, dateend, field2, exp=exp, coarse=coarse)

    da = (da1*da2).mean('time', keep_attrs=False)

    output_name = f'/archive/tlh/pp_xshield/20191020.00Z.C3072.L79x2_pire{exp}/monthly/{month}.{field1}-{field2}.nc'

    return da, output_name

if __name__ == "__main__":
    for exp in ['', '_PLUS_4K', '_CO2_1270ppmv', '_PLUS_4K_CO2_1270ppmv']:
        for month in shield_months():
            da, output_name = shield_monthly_mean_variance(month, field1='vs_coarse', field2='vs_coarse', exp=exp, coarse=True)
            da.to_dataset().to_netcdf(output_name)
