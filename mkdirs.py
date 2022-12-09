import os
from pathlib import Path
path = '/home/cwseitz/mnt/gpu/Data/'
path2 = '/home/cwseitz/mnt/gpu/Analysis/'
dirs = next(os.walk(path))[1]
for dir in dirs:
    Path(path2+dir).mkdir(parents=True, exist_ok=True)
