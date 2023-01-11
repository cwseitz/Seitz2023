from pipeline import Pipeline
from count import SpotCounts
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

prefixes = [
#'221130-Hela-IFNG-2h-1_1',
#'221218-Hela-IFNG-4h-1_1',
#'221130-Hela-IFNG-8h-1_1',
#'221130-Hela-IFNG-8h-2_1',
#'221130-Hela-IFNG-8h-3_1',
#'221130-Hela-IFNG-8h-4_1',
#'221218-Hela-IFNG-16h-2_1',
#'221206-Hela-IFNG-1h-1_1',
#'230105-Hela-IFNG-8h-1_2',
#'230107-Hela-IFNG-1h-1_2',
#'230107-Hela-IFNG-2h-1_1'
'230110-Hela-IFNG-2h-3_1',
'230110-Hela-IFNG-2h-4_1'
]


with open('config.json', 'r') as f:
    config = json.load(f)
    
for prefix in prefixes:
    print("Processing " + prefix)
    pipe = Pipeline(config,prefix)
    #pipe.tile()
    pipe.volume_contrast()
    #pipe.basic_correct()
    #pipe.apply_nucleus_model()
    #pipe.apply_cell_model()
    #pipe.segment_cells()

