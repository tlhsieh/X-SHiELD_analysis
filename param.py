from util import datenum2txt

def shield_dates(txt=False):
    dates = [
            ('20191101', '20200131'), 
            ('20201101', '20210131'), 
            ('20211101', '20220112')
    ]

    if txt:
        return [(datenum2txt(date[0], day=True), datenum2txt(date[1], day=True)) for date in dates]
    else:
        return dates

def shield_months():
    strings = []
    for yr in range(2019, 2021+1):
        for mo in range(1, 12+1):
            strings.append(f'{yr}{mo:02d}')

    return strings[10:]

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
        xlim = (-127, -105)
        ylim = (30, 50)
    elif domain_name == 'PNW':
        xlim = (-125, -115)
        ylim = (40, 50)
    elif domain_name == 'WA-OR-CA':
        xlim = (-125, -117)
        ylim = (33, 49)
    elif domain_name == 'WA-OR-CA_ocean':
        xlim = (-130, -117)
        ylim = (33, 49)
    elif domain_name == 'WA-OR':
        xlim = (-125, -117)
        ylim = (42, 49)
    elif domain_name == 'CA-NV':
        xlim = (-125, -115)
        ylim = (33, 42)
    elif domain_name == 'Sierra_Nevada':
        xlim = (-120, -118)
        ylim = (36, 39)
    elif domain_name == 'Alaska':
        xlim = (-170, -130)
        ylim = (50, 75)
    elif domain_name == 'Western':
        xlim = (-127, -110)
        ylim = (30, 50)
    else:
        print('Warning: domain_name not valid')
        xlim = (-180, 180)
        ylim = (-90, 90)

    return xlim, ylim

if __name__ == '__main__':
    print(shield_dates(txt=False))