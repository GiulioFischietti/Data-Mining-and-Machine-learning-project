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
trump_peaks = ["2020-10-05", "2020-10-05", "2020-10-12", "2020-11-07", "2020-11-16", "2020-12-12", "2021-01-06"]

# dates increased by one so that I can train again the model after the total occurrence of the event
trump_peaks_split = ["2020-09-01", "2020-10-06", "2020-10-13", "2020-11-08", "2020-11-17", "2020-12-13", "2021-01-07", "2021-01-08"]

# train the initial model  with the static df_n
static_df = pd.read_csv("./Tweets/samples/Trump/trump_set_sample.csv", sep=",", encoding = 'latin1', error_bad_lines=False)

X_train = static_df['tweet'].to_numpy()
y_train = static_df['class'].to_numpy()

stopWords = set(stopwords.words('english'))
stopWords = stemming_tokenizer(str(stopWords))

tfidf = TfidfVectorizer(analyzer='word', strip_accents = 'unicode', tokenizer = stemming_tokenizer, max_features = 10000, ngram_range = (1, 3), norm = 'l2')
X_train_tfidf = tfidf.fit_transform(raw_documents = X_train)
clf = svm.SVC(C = 1, gamma = 0.001, kernel = 'linear').fit(X_train_tfidf, y_train)


slide_df = static_df.copy()

for index, filepath in enumerate(glob.glob('./Tweets/preprocessed/Tweets to Trump/**/*.csv', recursive=True)):
    df_n = pd.read_csv(filepath, sep=",", encoding = 'latin1', error_bad_lines=False)
    df_n.dropna()
    print("Working on " + filepath)

    print("Trasforming tweets...")
    y_tweets = df_n['tweet'].to_numpy()
    y_tweets_tfidf = tfidf.transform(y_tweets)

    print("Predicting")
    y_pred = pd.DataFrame(data = clf.predict(y_tweets_tfidf), columns=["class"])
    
    print(y_pred.info())
    print()
    
    # write the results on a csv
    result = pd.concat([df_n['user_id'], df_n['conversation_id'], df_n['date'], y_pred['class']], axis=1)
    result.to_csv("./Tweets/results/Trump/"+ str(index)+ " results.csv", index = False)
