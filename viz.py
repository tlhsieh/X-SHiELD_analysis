import numpy as np
import xarray as xr
import geopandas
import matplotlib.cbook as cbook

def _get_world():
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    world = world.to_crs('EPSG:4326')
    return world
    
def _get_states():
    states = geopandas.read_file('/work/tlh/geopandas_data/usa-states-census-2014.shp')
    states = states.to_crs('EPSG:4326')
    return states

def plot_states(ax, plot_world=True):
    """
    lon range needs to be -180 to 180
    Example: plot_states(plt.gca())
    """
    
    if plot_world:
        _get_world().boundary.plot(ax=ax, colors='k', linewidths=0.5)
    _get_states().boundary.plot(ax=ax, colors='k', linewidths=0.5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

def plot_box(ax, limits, lw=3, ls='--', c='c'):
    xlim = limits[0]
    ylim = limits[1]

    ax.plot([xlim[0], xlim[0]], [ylim[0], ylim[1]], lw=lw, ls=ls, c=c)
    ax.plot([xlim[1], xlim[1]], [ylim[0], ylim[1]], lw=lw, ls=ls, c=c)
    ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[0]], lw=lw, ls=ls, c=c)
    ax.plot([xlim[0], xlim[1]], [ylim[1], ylim[1]], lw=lw, ls=ls, c=c)

def plot_box_whisker(ax, data, position, c='k', lw=1, label='', whis=(5, 95)):
    stats = cbook.boxplot_stats(data, whis=whis) # whis=(0, 100) shows min and max
    ax.bxp(stats, positions=[position], boxprops={'color': c, 'linewidth': lw}, whiskerprops={'color': c, 'linewidth': lw}, capprops={'color': c, 'linewidth': lw}, medianprops={'color': c, 'linewidth': lw, 'label': label}, showfliers=False) # showfliers is redundant when whis=(0, 100)

def _find_jump(array1d):
    """return the index at which array1d jumps from 0 to 360 or from 360 to 0"""
    
    return np.argmax(abs(np.diff(array1d)) > 350) + 1

def plot_tile_boundary(ax, color='k', transform=None):
    """Example: plot_tile_boundary(ax, transform=ccrs.PlateCarree())"""
    
    if transform is None:
        transform = ax.transData
        
    ds = xr.concat([xr.open_dataset(f'/archive/kyc/Stellar/20191020.00Z.C3072.L79x2_pire/history/2019102000/grid_spec_coarse.tile{i}.nc') for i in range(1, 6+1)], dim='tile')

    ax.plot(ds['grid_lont_coarse'][1][:, 0], ds['grid_latt_coarse'][1][:, 0], c=color, transform=transform)
    ax.plot(ds['grid_lont_coarse'][1][:, -1], ds['grid_latt_coarse'][1][:, -1], c=color, transform=transform)
    
    ax.plot(ds['grid_lont_coarse'][2][0, :], ds['grid_latt_coarse'][2][0, :], c=color, transform=transform)
    ax.plot(ds['grid_lont_coarse'][2][-1, :], ds['grid_latt_coarse'][2][-1, :], c=color, transform=transform)
    ax.plot(ds['grid_lont_coarse'][2][:, -1], ds['grid_latt_coarse'][2][:, -1], c=color, transform=transform)
    
    ax.plot(ds['grid_lont_coarse'][4][0, :], ds['grid_latt_coarse'][4][0, :], c=color, transform=transform)
    ax.plot(ds['grid_lont_coarse'][4][-1, :], ds['grid_latt_coarse'][4][-1, :], c=color, transform=transform)
    
    ax.plot(ds['grid_lont_coarse'][5][0, :], ds['grid_latt_coarse'][5][0, :], c=color, transform=transform)
    ax.plot(ds['grid_lont_coarse'][5][:, 0], ds['grid_latt_coarse'][5][:, 0], c=color, transform=transform)
    ax.plot(ds['grid_lont_coarse'][5][:, -1], ds['grid_latt_coarse'][5][:, -1], c=color, transform=transform)
    
    i_jump = _find_jump(ds['grid_lont_coarse'][2][:, 0])
    ax.plot(ds['grid_lont_coarse'][2][:i_jump, 0], ds['grid_latt_coarse'][2][:i_jump, 0], c=color, transform=transform)
    ax.plot(ds['grid_lont_coarse'][2][i_jump:, 0], ds['grid_latt_coarse'][2][i_jump:, 0], c=color, transform=transform)
    
    i_jump = _find_jump(ds['grid_lont_coarse'][5][-1, :])
    ax.plot(ds['grid_lont_coarse'][5][-1, :i_jump], ds['grid_latt_coarse'][5][-1, :i_jump], c=color, transform=transform)
    ax.plot(ds['grid_lont_coarse'][5][-1, i_jump:], ds['grid_latt_coarse'][5][-1, i_jump:], c=color, transform=transform)
    