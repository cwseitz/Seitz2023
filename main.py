from tiler import Tiler
from correct import Basic
from manual import Analyzer
from pathlib import Path

path0 = '/home/cwseitz/mnt/gpu/Data/'

paths = [
#'221206-Hela-IFNG-1h-1_1/', #GAPDH channel is not very good
#'221206-Hela-IFNG-1h-2_1/',
#'221206-Hela-IFNG-1h-3_2/',
#'221206-Hela-IFNG-1h-4_1/',
'221130-Hela-IFNG-2h-1_1/',
#'221130-Hela-IFNG-4h-3_1/',
#'221130-Hela-IFNG-4h-3_2/',
#'221130-Hela-IFNG-4h-4_1/',
#'221130-Hela-IFNG-8h-1_1/',
#'221130-Hela-IFNG-8h-2_1/',
#'221130-Hela-IFNG-8h-3_1/',
#'221130-Hela-IFNG-8h-4_1/'
#'221206-Hela-IFNG-16h-1_1/',
#'221206-Hela-IFNG-16h-2_1/',
#'221206-Hela-IFNG-16h-3_1/',
#'221206-Hela-IFNG-16h-4_1/',
#'221206-Hela-IFNG-24h-1_1/',
#'221206-Hela-IFNG-24h-3_1/',
#'221206-Hela-IFNG-24h-4_1/'
]

for i,path in enumerate(paths): 
    paths[i] = path0+paths[i]

def get_opath(ipath):
    return ipath.replace('Data', 'Analysis')
def get_prefix(ipath):
    return ipath.split('/')[-2]

for ipath in paths:
    prefix = get_prefix(ipath)
    print("Processing " + prefix)
    opath = get_opath(ipath)
    Path(opath).mkdir(parents=True, exist_ok=True)
    #tiler = Tiler(ipath,opath,prefix)
    #tiler.tile()
    #basic = Basic(opath,prefix)
    #basic.correct()
    analyzer = Analyzer(prefix,opath)
    analyzer.analyze()

