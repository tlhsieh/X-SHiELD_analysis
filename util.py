import numpy as np
import xarray as xr

import datetime

from scipy.interpolate import griddata

import sys
sys.path.append('/home/tlh/ipy')
from gfd import latlon, xrinterp

import glob

def datenum2txt(str_num, day=True):
    if day:
        date = datetime.datetime.strptime(str_num, '%Y%m%d')
        str_out = date.strftime('%Y%b%d')
    else:
        date = datetime.datetime.strptime(str_num, '%Y%m')
        str_out = date.strftime('%Y%b')
        
    return str_out

def date_linspace(beg_date, end_date, delta_day):
    """Return a list of date strings including end points"""
    
    dates = np.arange(datetime.datetime.strptime(beg_date, '%Y%m%d'), 
                      datetime.datetime.strptime(end_date, '%Y%m%d') + datetime.timedelta(days=1), 
                      datetime.timedelta(days=delta_day))
    
    return [date.astype(datetime.datetime).strftime('%Y%m%d') for date in dates]

def first_and_last_days_of_month(month='201911'):
    first_date = datetime.datetime.strptime(month, '%Y%m')
    first = first_date.strftime('%Y%m%d')
    last_date = (first_date + datetime.timedelta(days=31)).replace(day=1) - datetime.timedelta(days=1)
    last = last_date.strftime('%Y%m%d')

    return first, last

def _op_2d_to_3d(func2d, da3d):
    dim0 = da3d.dims[0]
    computed = [func2d(da3d[i]) for i in range(len(da3d[dim0]))]
    element = computed[0]
    
    return xr.DataArray(
        computed, 
        coords=[da3d[dim0], element[element.dims[-2]], element[element.dims[-1]]], 
        dims=[dim0, element.dims[-2], element.dims[-1]], 
        name=da3d.name
    )

def _op_2d_to_4d(func2d, da4d):
    dim0 = da4d.dims[0]
    dim1 = da4d.dims[1]
    computed = [[func2d(da4d[i, j]) for j in range(len(da4d[dim1]))] for i in range(len(da4d[dim0]))]
    element = computed[0][0]
    
    return xr.DataArray(
        computed, 
        coords=[da4d[dim0], da4d[dim1], element[element.dims[-2]], element[element.dims[-1]]], 
        dims=[dim0, dim1, element.dims[-2], element.dims[-1]], 
        name=da4d.name
    )

def op_2d_to_nd(func2d, da):
    ndim = len(da.dims)
    if ndim == 2:
        return func2d(da)
    elif ndim == 3:
        return _op_2d_to_3d(func2d, da)
    elif ndim == 4:
        return _op_2d_to_4d(func2d, da)
    else:
        print("ndim needs to be 2, 3 or 4")

def _roll_lon_2d(da):
    """from lon = 0-360 to lon = -180-180"""
    
    xvec = da[da.dims[-1]].values
    nx = len(xvec)
    
    if np.max(xvec) - np.min(xvec) < 350: # do not apply to regional data
        return da
    
    if np.max(xvec) < 180: # no need to roll
        return da
    
    da_out = xr.DataArray(np.roll(da.values, nx//2, axis=-1), 
                       coords=[da[da.dims[-2]], np.concatenate([xvec[nx//2:] - 360, xvec[:nx//2]])], 
                       dims=[da.dims[-2], da.dims[-1]])
    
    try: # preserve long_name
        da_out.attrs['long_name'] = da.attrs['long_name']
    except KeyError:
        pass
    
    return da_out

def roll_lon(da):
    """from lon = 0-360 to lon = -180-180"""

    return op_2d_to_nd(_roll_lon_2d, da)

def _flip_y_2d(da):
    da_out = xr.DataArray(np.flip(da.values, axis=-2), coords=[np.flip(da[da.dims[-2]].values), da[da.dims[-1]]], dims=da.dims)

    return da_out

def flip_y(da):
    return op_2d_to_nd(_flip_y_2d, da)

def sph2cart(da):
    """da is 2d on a sphere such as StageIV data"""
    
    xname = list(da.coords)[1]
    yname = list(da.coords)[0]
    xx = da[xname].values.flatten()
    yy = da[yname].values.flatten()
    dx = xx[1] - xx[0]
    dy = yy[1] - yy[0]

    points = np.transpose([xx, yy])
    values = da.values.flatten()

    xx_new = np.arange(np.min(xx), np.max(xx), dx)
    yy_new = np.arange(np.min(yy), np.max(yy), dy)

    da_interp = griddata(points, values, tuple(np.meshgrid(xx_new, yy_new)), method='linear')

    da_interp = xr.DataArray(da_interp, coords=[yy_new, xx_new], dims=['lat', 'lon'], name=da.name)

    try:
        da_interp.attrs['long_name'] = da.attrs['long_name']
    except KeyError:
        pass
    
    return da_interp

def crop(da, xlim, ylim):
    """Crop the given DataArray to the given boundaries; handels different lon conventions
    """

    xname = da.dims[-1]
    yname = da.dims[-2]

    if xlim[0] < da[xname].values[0]:
        da = roll_lon(da)

    return da.sel({xname: slice(xlim[0], xlim[1]), yname: slice(ylim[0], ylim[1])})

def find_corrupted_files(exp='', field='tsfc_coarse'):
    list1 = glob.glob(f'/archive/kyc/Stellar/20191020.00Z.C3072.L79x2_pire{exp}/history/*/{field}*.nc', recursive=True)
    list2 = glob.glob(f'/archive/kyc/Stellar_new/20191020.00Z.C3072.L79x2_pire{exp}/history/*/{field}*.nc', recursive=True)

    print('===== Corrupted files =====')
    for path in list1+list2:
        print(path)
        try:
            ds = xr.open_dataset(path)
        except OSError:
            print('???'+path)
    print('===== End =====')

if __name__ == "__main__":
    find_corrupted_files('_PLUS_4K', 'tsfc_coarse')