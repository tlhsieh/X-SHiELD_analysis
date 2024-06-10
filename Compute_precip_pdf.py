import numpy as np
import pandas as pd
import xarray as xr
# import matplotlib.pyplot as plt # optional
# plt.rc('font',**{'size':18})

from load import load_land, load_hgt, load_stage4, load_shield, load_spear_daily, load_am4_8xdaily
## load_stage4 requires Nio
from param import boundaries, shield_dates
from util import datenum2txt, roll_lon, op_2d_to_nd, crop, xrinterp, sph2cart
# from viz import plot_states # optional, requires cartopy

def roll_land_regional(da2d):
    """requires target_land (boolean land yes/no data on target grid)"""

    xlim, ylim = boundaries(domain_name)
    da_regional = crop(xrinterp(roll_lon(da2d), target_land).where(target_land), xlim, ylim)

    return da_regional

def get_pdf(da):
    da1d = da.values.flatten()
    da1d = da1d[np.isfinite(da1d)] # remove nan
    count, edges = np.histogram(da1d, bins=201, range=(-1, 401), density=False)
    count = xr.DataArray(count, coords=[(edges[:-1] + edges[1:])/2], dims=['rain'], name='count')

    return count, len(da1d)

def stage():
    da = load_stage4(datebeg, dateend)
    
    stage_interp = op_2d_to_nd(sph2cart, da)

    ## coarse grain data
    da_regional = op_2d_to_nd(roll_land_regional, stage_interp)

    ## compute pdf
    count, total_count = get_pdf(da_regional)

    return count, total_count, da_regional

def shield():
    da = load_shield(datebeg, dateend, field='pr', exp=exp, coarse=False)

    ## coarse grain data
    da_regional = op_2d_to_nd(roll_land_regional, da)
    da_regional = da_regional.resample(time='1D').mean()

    ## compute pdf
    count, total_count = get_pdf(da_regional)

    return count, total_count, da_regional

def spear():
    ens_range = range(1, 15+1)
    ens_dim = pd.Index(ens_range, name='ens')

    da = xr.concat([load_spear_daily(datebeg, dateend, lead=1, ens='%02d'%ens, field='precip') for ens in ens_range], dim=ens_dim)

    ## coarse grain data
    da_regional = op_2d_to_nd(roll_land_regional, da)

    ## compute pdf
    count_all = []
    total_count_all = []
    for ens in ens_range:
        count, total_count = get_pdf(da_regional.sel(ens=ens))
        count_all.append(count)
        total_count_all.append(xr.DataArray(total_count))

    return xr.concat(count_all, dim=ens_dim), xr.concat(total_count_all, dim=ens_dim), da_regional.isel(ens=0)

def am4():
    da = load_am4_8xdaily([1, 11, 12], yrbeg=11, yrend=20, field='pr', exp=exp)
    
    ## coarse grain data
    da_regional = op_2d_to_nd(roll_land_regional, da)
    da_regional = da_regional.resample(time='1D').mean()

    ## compute pdf
    count, total_count = get_pdf(da_regional)

    return count, total_count, da_regional

if __name__ == "__main__":
    ##### parameter list #####

    # model = 'StageIV'
    model = 'X-SHiELD'
    # model = 'SPEAR-med'
    # model = 'AM4'
    
    # target_grid = 'SPEAR-med'
    target_grid = 'X-SHiELD'
    
    exp = ''
    # exp = '_PLUS_4K'
    # exp = '_CO2_1270ppmv'
    # exp = '_PLUS_4K_CO2_1270ppmv'
    # exp = '_p4K'

    domain_name = 'WA-OR-CA'; land = True
    # domain_name = 'WA-OR-CA_ocean'; land = False

    ##### do the work #####

    # (datebeg, dateend) = ('20191201', '20191215')
    # for i in [0]:
    for i in range(3):
        (datebeg, dateend) = shield_dates()[i] #i

        if target_grid == 'SPEAR-med':
            target_land = roll_lon(load_land()) > 0.3 # SPEAR-med's land_frac
        else: # if target_grid == 'X-SHiELD'
            if model in ['StageIV', 'X-SHiELD']:
                target_land = roll_lon(load_hgt()) > 0 # X-SHiELD's height
            else: # if model == SPEAR-med or AM4
                target_land = roll_lon(load_land()) > 0.3 # SPEAR-med's land_frac
                print('Warning: invalid model for this target_grid')

        if land == False:
            target_land = ~target_land

        if model == 'StageIV':
            count, total_count, da_regional = stage()
        elif model == 'X-SHiELD':
            count, total_count, da_regional = shield()
        elif model == 'SPEAR-med':
            count, total_count, da_regional = spear()
        elif model == 'AM4':
            count, total_count, da_regional = am4()
        else:
            print('Error: invalid model name')

        if model == 'AM4':
            output_name = f'ar_data/daily_precip_pdf/precip_pdf_{target_grid}-grid_AM4{exp}_{domain_name}_Nov-Jan.nc'
        else:
            output_name = f'ar_data/daily_precip_pdf/precip_pdf_{target_grid}-grid_{model}{exp}_{domain_name}_{datenum2txt(datebeg)}-{datenum2txt(dateend)}.nc'

        ## save pdf
        ds = xr.Dataset()
        ds['counts'] = count # ds.count is a built-in method
        ds['total_counts'] = total_count
        ds.to_netcdf(output_name)

        # ## plot pdf
        # fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharey=True, facecolor='w')
        # ds['counts'].plot()
        # plt.yscale('log')
        # plt.savefig('pdf_test_1.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # ## map of interpolated, land-only data
        # fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharey=True, facecolor='w')
        # da_regional.isel(time=da_regional.max([da_regional.dims[-1], da_regional.dims[-2]]).argmax('time')).plot(vmin=0, vmax=60, cmap='Blues')
        # plot_states(ax, plot_world=False)
        # plt.savefig('pdf_test_2.png', dpi=300, bbox_inches='tight')
        # plt.close()