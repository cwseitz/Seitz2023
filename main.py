from tiler import Tiler
from correct import Basic
from manual import Analyzer

paths = [
'/research3/shared/cwseitz/Data/221206-Hela-IFNG-1h-1_1/',
#'/research3/shared/cwseitz/Data/221206-Hela-IFNG-1h-2_1/',
#'/research3/shared/cwseitz/Data/221206-Hela-IFNG-1h-3_2/',
#'/research3/shared/cwseitz/Data/221206-Hela-IFNG-1h-4_1/',
#'/research3/shared/cwseitz/Data/221130-Hela-IFNG-2h-1_1/',
#'/research3/shared/cwseitz/Data/221130-Hela-IFNG-4h-3_1/',
#'/research3/shared/cwseitz/Data/221130-Hela-IFNG-4h-3_2/',
#'/research3/shared/cwseitz/Data/221130-Hela-IFNG-4h-4_1/',
#'/research3/shared/cwseitz/Data/221130-Hela-IFNG-8h-1_1/',
#'/research3/shared/cwseitz/Data/221130-Hela-IFNG-8h-2_1/',
#'/research3/shared/cwseitz/Data/221130-Hela-IFNG-8h-3_1/',
#'/research3/shared/cwseitz/Data/221130-Hela-IFNG-8h-4_1/'
#'/research3/shared/cwseitz/Data/221206-Hela-IFNG-16h-1_1/',
#'/research3/shared/cwseitz/Data/221206-Hela-IFNG-16h-2_1/',
#'/research3/shared/cwseitz/Data/221206-Hela-IFNG-16h-3_1/',
#'/research3/shared/cwseitz/Data/221206-Hela-IFNG-16h-4_1/',
#'/research3/shared/cwseitz/Data/221206-Hela-IFNG-24h-1_1/',
#'/research3/shared/cwseitz/Data/221206-Hela-IFNG-24h-3_1/',
#'/research3/shared/cwseitz/Data/221206-Hela-IFNG-24h-4_1/'
]

def get_opath(ipath):
    return ipath.replace('Data', 'Analysis')
def get_prefix(ipath):
    return ipath.split('/')[-2]

filters = {
'min_area': 10000,
'max_area': 100000,
'max_ecc': 1,
'min_solid': 0
}


for ipath in paths:
    prefix = get_prefix(ipath)
    print("Processing " + prefix)
    opath = get_opath(ipath)
    #tiler = Tiler(ipath,opath,prefix)
    #tiler.tile()
    #basic = Basic(opath,prefix)
    #basic.correct()
    analyzer = Analyzer(prefix,opath)
    analyzer.analyze(filters)

