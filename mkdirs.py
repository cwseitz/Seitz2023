import os
from pathlib import Path
path = '/research3/shared/cwseitz/Data/'
path2 = '/research3/shared/cwseitz/Analysis/'
dirs = next(os.walk(path))[1]
for dir in dirs:
    Path(path2+dir).mkdir(parents=True, exist_ok=True)
