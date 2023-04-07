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

def plot_box(xrange, yrange, ax, lw=3, ls='--', c='c'):
    x1 = xrange[0]
    x2 = xrange[-1]
    y1 = yrange[0]
    y2 = yrange[-1]
    ax.plot([x1, x1], [y1, y2], lw=lw, ls=ls, c=c)
    ax.plot([x2, x2], [y1, y2], lw=lw, ls=ls, c=c)
    ax.plot([x1, x2], [y1, y1], lw=lw, ls=ls, c=c)
    ax.plot([x1, x2], [y2, y2], lw=lw, ls=ls, c=c)