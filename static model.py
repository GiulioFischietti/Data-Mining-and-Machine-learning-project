from collections import Counter
import numpy as np
from numpy.core.shape_base import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.svm import LinearSVC
import pprint
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression, PassiveAggressiveClassifier
from sklearn.model_selection import cross_val_predict
from textblob import TextBlob, tokenizers
from nltk.corpus import stopwords
from utils import stemming_tokenizer
from sklearn import svm
from sklearn.metrics import roc_auc_score, auc,roc_curve
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate, cross_val_predict, GridSearchCV
import pickle
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

df_dataset = pd.read_csv("./Tweets/samples/Biden/biden_set_sample.csv", sep=",", encoding = 'latin1', error_bad_lines=False)
df_dataset2 = pd.read_csv("./Tweets/samples/Trump/trump_set_sample.csv", sep=",", encoding = 'latin1', error_bad_lines=False)

df_dataset = df_dataset.dropna()
df_dataset2 = df_dataset2.dropna()

result = [df_dataset2, df_dataset]
df_dataset = pd.concat(result)

X_train = df_dataset2['tweet'].to_numpy()
y_train = df_dataset2['class'].to_numpy()


stopWords = set(stopwords.words('english'))

stopWords = stemming_tokenizer(str(stopWords))

# tfidf = TfidfVectorizer(analyzer='word', strip_accents = 'unicode', tokenizer = stemming_tokenizer, max_features = 5000, ngram_range = (1, 3), norm = 'l2')

# X_train_tfidf = tfidf.fit_transform(raw_documents=X_train)

# pickle.dump(tfidf, open("biden_tfidf.pickle", 'wb'))

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='word', strip_accents = 'unicode', tokenizer = stemming_tokenizer)),
    ('clf', PassiveAggressiveClassifier()),
])

# clf = svm.SVC(C = 1, gamma = 0.001, kernel = 'linear', class_weight="balanced").fit(X_train_tfidf, y_train)

# grid search for svc
# parameters = [  
#     {
#         'clf__C': [1, 10, 100, 1000], 
#         'clf__kernel': ['linear', 'rbf'], 
#         'clf__gamma': [0.001, 0.0001],
#         'tfidf__max_features': (500, 1000, 2000, 5000, 10000, 20000),
#         'tfidf__ngram_range': ((1, 1), (1, 2), (1,3)),
#         'tfidf__norm': ('l1', 'l2')
#     } 
#  ]

# grid search for MultinomialNB
# parameters = {
#     'clf__alpha': np.linspace(0.5, 1.5, 6),
#     'clf__fit_prior': [True, False],  
#     'tfidf__max_features': (500, 1000, 2000, 5000, 10000, 20000),
#     'tfidf__ngram_range': ((1, 1), (1, 2), (1,3)),
#     'tfidf__norm': ('l1', 'l2'),
# }

# grid search parameters for passive aggressive classifier
parameters={
    "clf__C": np.logspace(-3,3,7), 
    "clf__tol": [1e-3, 1e-4, 1e-5],
    "clf__loss": ['hinge', "squared_hinge"],
    'tfidf__max_features': (500, 1000, 2000, 5000, 10000, 20000),
    'tfidf__ngram_range': ((1, 1), (1, 2), (1,3)),
    'tfidf__norm': ('l1', 'l2'),
}

# grid search parameters for logistic regression classifier
# parameters={
#     "clf__C": np.logspace(-3,3,7), 
#     "clf__penalty": ['none', "l1", "elasticnet", "l2"],
#     'tfidf__max_features': (500, 1000, 2000, 5000, 10000, 20000),
#     'tfidf__ngram_range': ((1, 1), (1, 2), (1,3)),
#     'tfidf__norm': ('l1', 'l2'),
# }





# clf.fit(X_train_resampled, y_train_resampled)
# clf = svm.SVC(C = 100, gamma = 0.001, kernel = 'linear', class_weight = 'balanced').fit(X_train_tfidf, y_train)
# clf = SGDClassifier(alpha = 1e-06, max_iter  = 10, penalty = 'l2').fit(X_train_tfidf, y_train)
# clf = AdaBoostClassifier(n_estimators=100).fit(X_train_tfidf, y_train)
# clf = LogisticRegression(C = 1000.0, penalty = 'l2', class_weight = 'balanced').fit(X_train_tfidf, y_train)
# clf = PassiveAggressiveClassifier(C = 10.0, loss = 'hinge', tol = 0.001).fit(X_train_tfidf, y_train)
# clf = MultinomialNB(alpha = 0.5, fit_prior = False).fit(X_train_tfidf, y_train)
# y_pred = cross_val_predict(clf, X_train_tfidf, y_train, cv=7)

# print(metrics.classification_report(y_pred, y_train, target_names=['Contrary', 'Pro']))

# # pickle.dump(clf, open("biden_model.pickle", 'wb'))

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring = 'f1_weighted', cv = 5)
# print(grid_search.estimator.get_params())
print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint.pprint(parameters)
t0 = time.time()
grid_search.fit(X_train, y_train)
print("done in %0.3fs" % (time.time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
print(grid_search.best_params_)