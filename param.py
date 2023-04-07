def boundaries(domain_name=''):
    """Return a list of boundaries for the given domain_name.
    """

    if domain_name == '':
        xlim = (-180, 180)
        ylim = (-90, 90)
    elif domain_name == 'W_NA':
        xlim = (-150, -110)
        ylim = (30, 65)
    elif domain_name == 'W_US':
        xlim = (-130, -105)
        ylim = (30, 55)
    elif domain_name == 'PNW':
        xlim = (-125, -115)
        ylim = (40, 50)
    elif domain_name == 'CA-NV':
        xlim = (-125, -115)
        ylim = (33, 42)
    elif domain_name == 'Sierra_Nevada':
        xlim = (-120, -118)
        ylim = (36, 39)
    elif domain_name == 'Alaska':
        xlim = (-170, -130)
        ylim = (50, 75)
    else:
        xlim = (-127, -110)
        ylim = (32, 50)

    return xlim, ylim