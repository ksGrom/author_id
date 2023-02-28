from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from multiprocessing import Pool
from scipy.sparse import hstack
from tqdm import tqdm
import pandas as pd
import copy
import re


def check_work_titles_uniqueness(df, raise_exception=True):
    max_unique_n = df.groupby('work_title')['author'].describe().unique.max()
    if raise_exception and max_unique_n > 1:
        raise ValueError(f"Must be one author per title!")
    else:
        return max_unique_n <= 1


def _fit_predict_for_pool(clf_train_test: tuple):
    clf, X_train, X_test, y_train, y_test = clf_train_test
    clf.fit(X_train, y_train)
    y_pred_test = pd.Series(clf.predict(X_test), index=y_test.index)
    return clf, y_pred_test


def drop_work_title(df, work_title):
    mask = (df.work_title == work_title)
    df_test = df[mask]
    df_train = df[~mask]
    X_test, y_test = df_test.drop('author', axis=1), df_test.author
    X_train, y_train = df_train.drop('author', axis=1), df_train.author
    return X_train, X_test, y_train, y_test


def leave_one_title_out_cv_predictions(clf, df, n_jobs=6):
    dataset_list = []
    for work_title in df.work_title.unique():
        X_train, X_test, y_train, y_test = drop_work_title(df, work_title)
        dataset_list.append((copy.deepcopy(clf), X_train, X_test, y_train, y_test))
    with Pool(n_jobs) as p:
        clf_and_prediction_list = list(tqdm(
            p.imap(_fit_predict_for_pool, dataset_list),
            total=len(dataset_list)
        ))
    predictions = pd.concat([i[1] for i in clf_and_prediction_list], axis=0)
    classifiers = [i[0] for i in clf_and_prediction_list]
    return predictions, classifiers


def leave_one_title_out_cv_score(
        df,
        *,
        clf=None,
        predictions=None,
        f1_average='micro'
):
    if (clf is not None and predictions is not None) or \
            (clf is None and predictions is None):
        raise ValueError("Either clf or predictions must be passed!")
    work_title_list = []
    score_list = []
    average_score = 0
    if clf is not None:
        predictions, _ = leave_one_title_out_cv_predictions(clf, df)
    for work_title in df.work_title.unique():
        X_train, X_test, y_train, y_test = drop_work_title(df, work_title)
        if df.author.unique().shape[0] <= 2:
            pos_label = y_test.iloc[0]
            f1_average = 'binary'
        else:
            pos_label = None
        score = f1_score(y_test, predictions[y_test.index],
                         pos_label=pos_label, average=f1_average)
        score_list.append(score)
        average_score += score * y_test.shape[0] / df.shape[0]
        work_title_list.append(work_title)
    return predictions, work_title_list, score_list, average_score


class CustomTfidfVectorizer:
    def __init__(self, **kwargs):
        self.__char_tfidf_params = None
        self.__lemma_tfidf_params = None
        self.__patterns = None

        self.char_tfidf_params = kwargs.get('char_tfidf_params')
        self.lemma_tfidf_params = kwargs.get('lemma_tfidf_params')
        self.patterns = kwargs.get('patterns')

        self.char_tfidf_vectorizer = TfidfVectorizer(**self.char_tfidf_params)
        self.lemma_tfidf_vectorizer = TfidfVectorizer(**self.lemma_tfidf_params)

    def fit(self, X, y=None):
        self.__fit_transform(X, 'fit')
        return self

    def transform(self, X, y=None):
        return hstack(self.__fit_transform(X, 'transform'))

    def fit_transform(self, X, y=None):
        return hstack(self.__fit_transform(X, 'fit_transform'))

    def __fit_transform(self, X, method_name):
        X_text, X_lemmas = self.__get_text_and_lemmas_series(X)
        X_text_res = getattr(self.char_tfidf_vectorizer, method_name)(X_text)
        X_lemmas_res = getattr(self.lemma_tfidf_vectorizer, method_name)(X_lemmas)
        return X_text_res, X_lemmas_res

    def __get_text_and_lemmas_series(self, pandas_obj):
        if not (type(pandas_obj) is pd.DataFrame
                and hasattr(pandas_obj, 'text')
                and hasattr(pandas_obj, 'lemmas')):
            raise ValueError("X must be pandas.DataFrame with 'text' and 'lemmas' columns!")
        text_series = pandas_obj.text.apply(
            lambda x: re.sub(self.patterns, '', x.lower()))
        lemmas_series = pandas_obj.lemmas
        if text_series.isna().sum() + lemmas_series.isna().sum() > 0:
            raise ValueError("Nan value(s) in 'text' or/and 'lemmas' columns!")
        return text_series, lemmas_series

    @staticmethod
    def __replace_punctuation_marks(text):
        replacements = {
            "...": "…",
            "–": "—",
            "«": "\"",
            "»": "\"",
            "\n": " "
        }
        for key, val in replacements.items():
            text = text.replace(key, val)
        text = re.sub(' +', ' ', text)
        return text

    @property
    def patterns(self):
        return self.__patterns

    @patterns.setter
    def patterns(self, value):
        self.__patterns = r'[^а-яё.?!…,:;—()\"\'\-\n ]+' \
            if value is None else value

    @property
    def lemma_tfidf_params(self):
        return self.__lemma_tfidf_params

    @lemma_tfidf_params.setter
    def lemma_tfidf_params(self, value):
        self.__lemma_tfidf_params = {
            'analyzer': 'word',
            'ngram_range': (1, 1),
            'sublinear_tf': True,
            'max_features': 10000,
        }
        if type(value) is dict:
            self.__lemma_tfidf_params.update(value)

    @property
    def char_tfidf_params(self):
        return self.__char_tfidf_params

    @char_tfidf_params.setter
    def char_tfidf_params(self, value):
        self.__char_tfidf_params = {
            'analyzer': 'char',
            'ngram_range': (1, 4),
            'sublinear_tf': True,
            'max_features': 5000,
        }
        if type(value) is dict:
            self.__char_tfidf_params.update(value)


class AuthorIdentificationTfidfPipeline:
    def __init__(self, **kwargs):
        self.__sgd_args = None

        self.sgd_classifier_args = kwargs.get('sgd_classifier_args')

        estimators = [
            ("tfidf", CustomTfidfVectorizer(**kwargs)),
            ("clf", SGDClassifier(**self.sgd_classifier_args))
        ]
        self.pipeline = Pipeline(estimators)

    def fit(self, X, y=None):
        if y is None:
            X, y = X.drop('author', axis=1), X.author
        check_work_titles_uniqueness(pd.concat([X, y], axis=1))
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    @property
    def sgd_classifier_args(self):
        return self.__sgd_args

    @sgd_classifier_args.setter
    def sgd_classifier_args(self, value):
        self.__sgd_args = {
            'random_state': 42,
            'max_iter': 50000,
            'verbose': 0,
            'n_jobs': -1,
            'loss': 'modified_huber'
        }
        if type(value) is dict:
            self.__sgd_args.update(value)
