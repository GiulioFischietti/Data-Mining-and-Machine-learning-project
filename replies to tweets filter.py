import pandas as pd
import glob

tweets_name = "./Tweets/original/tweets from biden.csv"

tweets = pd.read_csv(tweets_name, sep=",", encoding = 'utf_8', error_bad_lines=False)

# Filtering only replies to biden's posts... this still includes retweets, replies to retweets and replies to other users reples in biden's posts.

# for filepath in glob.iglob('./Tweets/preprocessed/Tweets to Trump/*.txt'):
for filepath in glob.iglob('./Tweets/preprocessed/Tweets to Biden/*.csv'):

    replies = pd.read_csv(filepath, sep=",", encoding = 'latin1', error_bad_lines=False)
    mask = replies[replies['conversation_id'].isin(tweets['id'])]
    mask.to_csv(path_or_buf = (filepath.replace(".csv", "") + " only_replies.csv"), index = False)

    print(mask.info())
    print(replies.info())
