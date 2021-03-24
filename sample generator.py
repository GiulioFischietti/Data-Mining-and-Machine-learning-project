import pandas as pd
import glob

from pandas.core.frame import DataFrame


df = []
# for filepath in glob.iglob('./Tweets/preprocessed/Tweets to Trump/*.txt'):
for filepath in glob.iglob('./Tweets/preprocessed/Tweets to Biden/set/*.csv'):
    df_n = pd.read_csv(filepath, sep=",", encoding = 'latin1', error_bad_lines=False)
    df.append(df_n)
    print(df_n.info())

result = pd.concat(df)

sample = result.sample(n = 1000)

sample.to_csv(path_or_buf = ("./Tweets/samples/Biden/biden_set_sample 2.csv"), index = False)