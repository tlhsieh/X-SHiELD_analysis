import numpy as np
import xarray as xr

import datetime

from scipy.interpolate import interp1d, griddata

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

    if len(element.dims) == 2:
        coords = [da3d[dim0], element[element.dims[-2]], element[element.dims[-1]]]
        dims = [dim0, element.dims[-2], element.dims[-1]]
    elif len(element.dims) == 1:
        coords = [da3d[dim0], element[element.dims[-1]]]
        dims = [dim0, element.dims[-1]]
    else:
        coords = [da3d[dim0]]
        dims = [dim0]
    
    return xr.DataArray(computed, coords=coords, dims=dims, name=da3d.name)

def _op_2d_to_4d(func2d, da4d):
    dim0 = da4d.dims[0]
    dim1 = da4d.dims[1]
    computed = [[func2d(da4d[i, j]) for j in range(len(da4d[dim1]))] for i in range(len(da4d[dim0]))]
    element = computed[0][0]

    if len(element.dims) == 2:
        coords = [da4d[dim0], da4d[dim1], element[element.dims[-2]], element[element.dims[-1]]]
        dims = [dim0, dim1, element.dims[-2], element.dims[-1]]
    elif len(element.dims) == 1:
        coords = [da4d[dim0], da4d[dim1], element[element.dims[-1]]]
        dims = [dim0, dim1, element.dims[-1]]
    else:
        coords = [da4d[dim0], da4d[dim1]]
        dims = [dim0, dim1]
    
    return xr.DataArray(computed, coords=coords, dims=dims, name=da4d.name)

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

def change_lon(lon):
    """change lon value; order not changed"""

    to_shift = lon > 180
    lon = lon - 360*to_shift
    
    return lon

def _change_lon_axis_1d(da):
    xname = da.dims[-1]

    return xr.DataArray(da.values, coords=[change_lon(da[xname].values)], dims=da.dims)

def _change_lon_axis_2d(da):
    xname = da.dims[-1]
    yname = da.dims[-2]

    return xr.DataArray(da.values, coords=[da[yname].values, change_lon(da[xname].values)], dims=da.dims)

def change_lon_axis(da):
    """Designed for regional data in the western hemisphere
    From lon = 0-360 to lon = -180-180
    Change the lon axis only, not the data
    """

    if len(da.dims) == 1:
        return _change_lon_axis_1d(da)

    return op_2d_to_nd(_change_lon_axis_2d, da)

def _roll_lon_1d(da):
    """from lon = 0-360 to lon = -180-180"""
    
    xvec = da[da.dims[-1]].values
    nx = len(xvec)
    
    if np.max(xvec) - np.min(xvec) < 350: # do not apply to regional data
        return da
    
    if np.max(xvec) < 180: # no need to roll
        return da
    
    da_out = xr.DataArray(np.roll(da.values, nx//2, axis=-1), 
                       coords=[np.concatenate([xvec[nx//2:] - 360, xvec[:nx//2]])], 
                       dims=[da.dims[-1]])
    
    try: # preserve long_name
        da_out.attrs['long_name'] = da.attrs['long_name']
    except KeyError:
        pass
    
    return da_out

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

    if len(da.dims) == 1:
        return _roll_lon_1d(da)

    return op_2d_to_nd(_roll_lon_2d, da)
    
def _flip_y_2d(da):
    da_out = xr.DataArray(np.flip(da.values, axis=-2), coords=[np.flip(da[da.dims[-2]].values), da[da.dims[-1]]], dims=da.dims)

    return da_out

def flip_y(da):
    return op_2d_to_nd(_flip_y_2d, da)

def exchange_xy(da, yvec=[]):
    """exchange the x and y coordinates of the input 1d data, after interpolating the original y axis to yvec
    """
    
    xname = da.dims[0]
    yname = da.name
    
    xx = da[xname].values
    yy = da.values
    
    ## remove nan
    xx = xx[np.isfinite(yy)]
    yy = yy[np.isfinite(yy)]

    if len(yvec) == 0:
        yvec = np.linspace(np.min(yy), np.max(yy), len(yy))
    
    xnew = interp1d(yy, xx, fill_value='extrapolate')(yvec)
    
    return xr.DataArray(xnew, coords=[yvec], dims=[yname], name=xname)

def pdf_to_percentile(pdf):
    """Given an 1d xr.DataArray p(x), compute CDF then return x(percentile)
    """
    
    if np.sum(pdf) < 0.99:
        print('Warning: sum of PDF should be 1')
        
    cdf = np.cumsum(pdf)
    cdf_exchanged = exchange_xy(cdf)
    
    xname = cdf_exchanged.dims[0]
    xvec = cdf_exchanged[xname]
    
    percentile = xvec*100
    
    return xr.DataArray(cdf_exchanged.values, coords=[percentile], dims=['Percentile'], name=cdf_exchanged.name)
    
def pdf_to_log10percentile(pdf, vmin=-5, vmax=-1):
    """Given an 1d xr.DataArray p(x), compute CDF then return x(log10(1 - percentile))
    This calculation prevents loss of data for extreme events. 
    
    E.g., vmin = -6 gives percentile value of 1 - 1e-5, vmax = -1 gives percentile value of 0.9 (i.e. 90th percentile)
    
    Note: don't convert the horizontal axis here, instead do it when plotting to preserve the log scale
        xticks = np.arange(-6, 0, 1.)
        plt.xticks(xticks, 100 - 100*10**xticks)
        plt.xticks(rotation=45)
        plt.xlabel('Percentile')
        plt.gca().invert_xaxis()
    """
    
    if np.sum(pdf) < 0.99:
        print('Warning: sum of PDF should be 1')
        
    cdf = np.cumsum(pdf)
    cdf_extreme = np.log10(1 - cdf) # change the vertical axis to emphasize the extreme events
    cdf_exchanged = exchange_xy(cdf_extreme, np.linspace(vmin, vmax, len(cdf)))
    
    return cdf_exchanged.rename({cdf.name: '1 - log10(perc)'})
    
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

    if xlim[0] < da[xname].values[0] and (da[xname].values[-1] - da[xname].values[0]) > 350: # if out of bounds can be fixed by roll_lon
        if xlim[1] < da[xname].values[0]: # if box is entirely in the western hemisphere
            da = da.sel({xname: slice(xlim[0]+360, xlim[1]+360), yname: slice(ylim[0], ylim[1])})
            da = change_lon_axis(da) # fast
        else: # if box crosses 0 deg
            da = roll_lon(da) # slow
            da = da.sel({xname: slice(xlim[0], xlim[1]), yname: slice(ylim[0], ylim[1])})
    else: # don't need to roll
        da = da.sel({xname: slice(xlim[0], xlim[1]), yname: slice(ylim[0], ylim[1])})

    return da
    
def bin_average(x, y, bins=10):
    """Return y(x), where y is averaged over each x bin"""
    
    count, edges = np.histogram(x, bins)
    accum, edges = np.histogram(x, bins, weights=y)
    
    x_avg = (edges[:-1] + edges[1:])/2
    y_avg = accum/count
    
    return x_avg, y_avg

def scatter_to_percentiles(xvec, yvec, edges=[], percentiles=[50]):
    """Percentile values over each x bin"""

    if len(edges) == 0:
        edges = np.linspace(np.nanmin(xvec), np.nanmax(xvec), 101)
        
    idxsort = np.argsort(xvec)
    xsorted = xvec[idxsort]
    ysorted = yvec[idxsort]
    
    idxedges = np.searchsorted(xsorted, edges)
    
    output = np.zeros((len(edges) - 1, len(percentiles)))
    
    for i in range(len(edges) - 1):
        subvec = ysorted[idxedges[i]:idxedges[i+1]]
        if len(subvec) > 0:
            output[i,:] = np.percentile(subvec, percentiles)
        else:
            output[i,:] = np.nan
            
        # output[i,:] = np.mean(subvec) # equivalent to bin_average()
        
    bins = (edges[:-1] + edges[1:])/2
        
    return bins, output

def nan2zero(da):
    nparray = da.values
    nparray[np.isnan(nparray)] = 0

    return xr.DataArray(nparray, coords=da.coords)

from scipy.interpolate import interp2d

def xrinterp(da, target):
    """interpolation to target's grid"""
    
    da = nan2zero(da) # some data have nan on land, which breaks interp2d
    npinterp = interp2d(da[da.dims[-1]], da[da.dims[-2]], da)(target[target.dims[-1]], target[target.dims[-2]])
    da2 = xr.DataArray(npinterp, coords=[target[target.dims[-2]], target[target.dims[-1]]], dims=[target.dims[-2], target.dims[-1]], name=da.name)
    
    return da2
    
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