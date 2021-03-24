import pandas as pd
import glob
from pandas.core.frame import DataFrame

df = []

# Load all the tweets

# for filepath in glob.glob('./Tweets/preprocessed/Tweets to Biden/**/*.csv', recursive=True):
for filepath in glob.glob('./Tweets/preprocessed/Tweets to Trump/**/*.csv', recursive=True):
    df_n = pd.read_csv(filepath, sep=",", encoding = 'latin1', error_bad_lines=False)
    df.append(df_n)

result = pd.concat(df)

trump_peaks = ["2020-10-05", "2020-10-12", "2020-11-07", "2020-11-16", "2020-12-12", "2021-01-06"]
# biden_peaks = ["2020-09-30", "2020-10-06", "2020-10-12", "2020-10-23", "2020-11-04", "2020-11-25", "2020-12-20", "2021-01-07"]

# filter the tweets by peak day events and get a random sample

for index, peak in enumerate(trump_peaks):

    peakTweets = result[result['date'] == peak]
    samplePeak = peakTweets.sample(n = 700)
    # samplePeak.to_csv(path_or_buf = ("./Tweets/samples/Biden/biden peak " + peak + '.csv'), index = False)
    samplePeak.to_csv(path_or_buf = ("./Tweets/samples/Trump/trump peak " + peak + '(2).csv'), index = False)
    print(samplePeak.info())
