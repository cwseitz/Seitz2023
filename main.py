from pipeline import Pipeline

prefixes = [
#'221206-Hela-IFNG-1h-1_1/',
#'221206-Hela-IFNG-1h-2_1/',
#'221206-Hela-IFNG-1h-3_2/',
#'221206-Hela-IFNG-1h-4_1/',
#'221130-Hela-IFNG-2h-1_1/',
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
'221218-Hela-IFNG-16h-2_1/'
#'221218-Hela-IFNG-4h-2_1/'
]

config = ConfigParser.from_args(args, options)
for prefix in prefixes:
    print("Processing " + prefix)
    Path(analpath+prefix).mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(config,prefix)
    pipe.execute()



