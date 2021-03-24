# incremental model:

# passo 1: caricare il dataset dei tweet del mese set
# passo 2: caricare i dataset corrispondenti ai giorni dei singoli picchi
# passo 3: effettuare il primo train sui 30 giorni di settembre
# passo 4: classificare i tweet del picco e vedere la matrice di confusione
# passo 5: aggiungere al dataset i tweet del picco i
# passo 6: effettuare il train e classificare i tweet del picco i+1, fare questo iterativamente per tutti i picchi

# eventualmente usare il modello nuovo di ogni iterazione per classificare tutti i picchi, se di prestazioni migliori usarlo per il risultato finale.


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import stemming_tokenizer
from nltk.corpus import stopwords
from sklearn import svm
import sklearn.metrics
from sklearn.model_selection import cross_val_predict
import time 
from sklearn.linear_model import PassiveAggressiveClassifier

df_dataset = pd.read_csv("./Tweets/samples/Trump/trump_set_sample.csv", sep = ",", encoding = 'latin1', error_bad_lines=False)
df_dataset = df_dataset.dropna()

X_train = df_dataset['tweet'].to_numpy()
y_train = df_dataset['class'].to_numpy()

stopWords = set(stopwords.words('english'))
stopWords = stemming_tokenizer(str(stopWords))

tfidf = TfidfVectorizer(analyzer='word', strip_accents = 'unicode', tokenizer = stemming_tokenizer, max_features = 10000, ngram_range = (1, 3), norm = 'l2')
tfidf_set = TfidfVectorizer(analyzer='word', strip_accents = 'unicode', tokenizer = stemming_tokenizer, max_features = 10000, ngram_range = (1, 3), norm = 'l2')
tfidf_sliding = TfidfVectorizer(analyzer='word', strip_accents = 'unicode', tokenizer = stemming_tokenizer, max_features = 10000, ngram_range = (1, 3), norm = 'l2')

X_train_tfidf = tfidf.fit_transform(raw_documents = X_train)
tfidf_set.fit_transform(raw_documents = X_train)
tfidf_sliding.fit_transform(raw_documents = X_train)

trump_peaks = ["2020-10-05", "2020-10-12", "2020-11-07", "2020-11-16", "2020-12-12", "2021-01-06"]

clf = svm.SVC(C = 1, gamma = 0.001, kernel = 'linear').fit(X_train_tfidf, y_train)
clf_set = svm.SVC(C = 1, gamma = 0.001, kernel = 'linear').fit(X_train_tfidf, y_train)
clf_sliding = svm.SVC(C = 1, gamma = 0.001, kernel = 'linear').fit(X_train_tfidf, y_train)

print("Dimension of dictionary: " + str(X_train_tfidf.shape))

# df_incremental = [df_dataset]
# df_window = [df_dataset]
new_df = df_dataset.copy()
slide_df = df_dataset.copy()

for index, peak in enumerate(trump_peaks):

    df_n = pd.read_csv("./Tweets/samples/Trump/trump peak " + peak + '.csv', sep = ",", encoding = 'latin1', error_bad_lines=False)
    
    X_test = df_n['tweet'].to_numpy()
    y_test = df_n['class'].to_numpy()

    X_test_tfidf = tfidf.transform(X_test)
    X_test_tfidf_old = tfidf_set.transform(X_test)
    X_test_tfidf_slide = tfidf_sliding.transform(X_test)

    y_pred = clf.predict(X_test_tfidf)
    y_pred_old = clf_set.predict(X_test_tfidf_old)
    y_pred_slide = clf_sliding.predict(X_test_tfidf_slide)

    print(peak)
    print('New model accuracy: ' + str(sklearn.metrics.f1_score(y_pred, y_test)))
    print('SW model accuracy: ' + str(sklearn.metrics.f1_score(y_pred_slide, y_test)))
    print('Original model accuracy: ' + str(sklearn.metrics.f1_score(y_pred_old, y_test)))
    print()
    print()
    # df_incremental.append(df_n)
    # df_window.append(df_n)

    start_time = time.time()
    new_df = new_df.append(df_n)
    # print(new_df.info())
    new_X_train = new_df['tweet'].to_numpy()
    new_y_train = new_df['class'].to_numpy()
    new_X_train_tfidf = tfidf.fit_transform(raw_documents = new_X_train)
    # new_df.to_csv(path_or_buf="incremental " + peak + ".csv", index = False)
    clf.fit(new_X_train_tfidf, new_y_train)
    elapsed_time = time.time() - start_time
    print("Elapsed time for incremental model train: " + str(elapsed_time))

    start_time = time.time()
        
    # print(slide_df.info())
    slide_df = slide_df.append(df_n)
    slide_df = slide_df[70:]
    slide_df.to_csv(path_or_buf = "sliding window " + peak + ".csv", index = False)
    slide_X_train = slide_df['tweet'].to_numpy()
    slide_y_train = slide_df['class'].to_numpy()
    slide_X_train_tfidf = tfidf_sliding.fit_transform(raw_documents=slide_X_train)
    

    clf_sliding.fit(slide_X_train_tfidf, slide_y_train)
    elapsed_time = time.time() - start_time
    print("Elapsed time for sliding window model train: " + str(elapsed_time))
    print()