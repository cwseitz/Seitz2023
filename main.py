from pipeline import Pipeline
import json

prefixes = [
'221130-Hela-IFNG-2h-1_1'
#'221218-Hela-IFNG-4h-1_1'
#'221130-Hela-IFNG-8h-1_1'
#'221218-Hela-IFNG-16h-2_1'
]

with open('config.json', 'r') as f:
    config = json.load(f)
for prefix in prefixes:
    print("Processing " + prefix)
    pipe = Pipeline(config,prefix)
    pipe.execute()




