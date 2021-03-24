import pandas as pd
import glob



# for filepath in glob.iglob('./Tweets/preprocessed/Tweets to Trump/*.txt'):
for filepath in glob.iglob('./Tweets/preprocessed/Tweets to Biden/*.csv'):

    replies = pd.read_csv(filepath, sep=",", encoding = 'latin1', error_bad_lines=False)
    result = replies[replies['tweet'].str.count("@")==1]
    
    print(replies.info())
    print(result.info())

    result.to_csv(path_or_buf = (filepath.replace(".csv", "") + " to biden.csv"), index = False)

