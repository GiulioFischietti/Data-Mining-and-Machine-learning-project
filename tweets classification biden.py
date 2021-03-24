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

# peak dates
biden_peaks = ["2020-09-30", "2020-10-06", "2020-10-12", "2020-10-23", "2020-11-04", "2020-11-25", "2020-12-20", "2021-01-07", "2021-01-08"]

# dates increased by one so that I can train again the model after the total occurrence of the event
biden_peaks_split = ["2020-09-01", "2020-10-01", "2020-10-07", "2020-10-13", "2020-10-24", "2020-11-05", "2020-11-26", "2020-12-21", "2021-01-08"]

# train the initial model  with the static dataset
static_df = pd.read_csv("./Tweets/samples/Biden/biden_set_sample.csv", sep=",", encoding = 'latin1', error_bad_lines=False)
static_df.dropna(inplace=True)
X_train = static_df['tweet'].to_numpy()
y_train = static_df['class'].to_numpy()

stopWords = set(stopwords.words('english'))
stopWords = stemming_tokenizer(str(stopWords))

tfidf = TfidfVectorizer(analyzer='word', strip_accents = 'unicode', tokenizer = stemming_tokenizer, max_features = 5000, ngram_range = (1, 3), norm = 'l1')
X_train_tfidf = tfidf.fit_transform(raw_documents = X_train)
print(y_train)
clf_sliding = svm.SVC(C = 100, gamma = 0.001, kernel = 'linear', class_weight="balanced").fit(X_train_tfidf, y_train)

dataset_df = []

#integrate the dataset
for filepath in glob.glob('./Tweets/preprocessed/Tweets to biden/**/*.csv', recursive=True):
    df_n = pd.read_csv(filepath, sep=",", encoding = 'latin1', error_bad_lines=False)
    df_n.dropna()
    dataset_df.append(df_n)
    
    # result = pd.concat([df_n['user_id'],df_n['conversation_id'],df_n['date'], tweets_pred['class']], axis=1)
    # print(result.info())
    # result.to_csv(path_or_buf="./Tweets/results/biden/" + filepath.replace("\\", "/").split("/")[-1].replace("decoded only_replies to biden", "results"), index=False)
    # print("./Tweets/results/" + filepath.replace("\\", "/").split("/")[-1].replace("decoded only_replies to biden", "results"))

dataset_df = pd.concat(dataset_df)
# dataset_df.info()

# split the dataset by peak dates
dataset_df['date'] = dataset_df['date'].astype('datetime64[ns]')
dataset_df = dataset_df.set_index(dataset_df['date'])
dataset_df = dataset_df.sort_index()

dataframe_split = []

for index, peak in enumerate(biden_peaks_split):
    if((index) < len(biden_peaks_split)):
        dataframe_split.append(dataset_df[peak: biden_peaks[index]])

del dataset_df

slide_df = static_df.copy()
# for each split, classify all the replies in it and then train again on the relative biden_peak
for index, dataset in enumerate(dataframe_split):
    print("Working on " + biden_peaks_split[index])

    print("Trasforming tweets...")
    y_tweets = dataset['tweet'].to_numpy()
    y_tweets_tfidf = tfidf.transform(y_tweets)

    print("Predicting")
    y_pred = pd.DataFrame(data = clf_sliding.predict(y_tweets_tfidf), columns=["class"])
    
    # print(y_pred.info())
    # print()
   
    # write the results on a csv
    print(dataset['date'])
    print(y_pred['class'])
    dataset = dataset.reset_index(drop = True)
    print(dataset.info())
    result = pd.concat([dataset["user_id"],dataset['conversation_id'],dataset['date'], y_pred['class']], axis=1)
    print(result.info())
    result.to_csv("./Tweets/results/Biden/"+biden_peaks_split[index]+" results.csv", index = False)

    if(index<(len(biden_peaks)-1)):
        df_i = pd.read_csv("./Tweets/samples/biden/"+ "biden peak " + biden_peaks[index] + ".csv", sep=",", encoding = 'latin1', error_bad_lines=False)
        slide_df = slide_df.append(df_i)
        slide_df = slide_df[70:]
        # print(slide_df.info())
        tweet_tfidf = tfidf.fit_transform(slide_df['tweet'].to_numpy())
        y_train = slide_df['class'].to_numpy()
        clf_sliding.fit(tweet_tfidf, y_train)

print("final")
print("Trasforming tweets...")
y_tweets = dataframe_split[-1]
y_tweets = y_tweets[y_tweets['date']=="2021-01-08"]
y_tweets = y_tweets['tweet']
y_tweets_tfidf = tfidf.transform(y_tweets)

print("Predicting")
y_pred = pd.DataFrame(data = clf_sliding.predict(y_tweets_tfidf), columns=["class"])

result = pd.concat([dataset["user_id"],dataset['conversation_id'],dataset['date'], y_pred['class']], axis=1)
print(result.info())
result.to_csv("./Tweets/results/Biden/2021-01-08 results.csv", index = False)