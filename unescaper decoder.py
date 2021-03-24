import html
import pandas
import glob
import os

for filepath in glob.iglob('./Tweets/preprocessed/Tweets to Biden/*.txt'):
    with open(filepath, encoding="latin1") as f, open(filepath.replace(".txt", "") + ' decoded.csv', 'w') as g:
        content = html.unescape(f.read().encode('ascii', 'ignore').decode('utf8'))
        g.write(content)
        print(filepath.replace(".csv", "") + ' decoded.csv')

# Deletes old files. be careful

for filepath in glob.iglob('./Tweets/preprocessed/Tweets to Biden/*.txt'):
    os.remove(filepath)