import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import stemming_tokenizer

data = []
for filepath in glob.glob('./Tweets/stance/Biden/*.csv', recursive=True):
    df_n = pd.read_csv(filepath, sep=",", encoding = 'latin1', error_bad_lines=False)
    data.append(df_n)
 
result = pd.concat(data)
# print(result.info())
# result.drop_duplicates(inplace=True)
# result.dropna(inplace=True)
result.to_csv("./Tweets/stance/biden/stance biden.csv", index = False)
# print(result.info())


# splitto il dataframe per eventi

# per ogni evento, aggiorno lo split i con gli user precedenti, eventualmente aggiornando stance

final_df = []

for event, df_event in result.groupby(['event']):
    # df_event.set_index(['user_id'])
    final_df.append(df_event)

merge_dataset = []

for index, df in enumerate(final_df):
    if(index > 0):
        tmp = df.merge(merge_dataset[index-1], how = "outer")
        tmp.drop_duplicates(inplace = True, subset = ['user_id'], keep = 'last')
        merge_dataset.append(tmp)
        merge_dataset[index]['event'] = np.full((len(merge_dataset[index]['event']), 1), index)
        merge_dataset[index].sort_values(by=['user_id'], inplace = True)
    else:
        merge_dataset.append(df)

    # merge_dataset[index].to_csv( "./Tweets/stance/Trump/incremental event " + str(index)+ ' users stance.csv', index = False)

merge_dataset = pd.concat(merge_dataset)
merge_dataset.to_csv('./Tweets/stance/Biden/incremental users stance.csv', index = False)