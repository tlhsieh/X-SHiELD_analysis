import geopandas

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