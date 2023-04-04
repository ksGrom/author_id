from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
from collections import Counter
import pandas as pd
import numpy as np
import re

import dataset_utils as ds
EXCERPT_LEN = 250


def check_work_titles_uniqueness(df: pd.DataFrame, raise_exception=True):
    """Проверяет наличие в датасете неуникальных названий
    произведений: если есть два автора с одноименными произведениями,
    возвращает `False` или поднимает `ValueError` (в зависимости от
    аргумента `raise_exception`).
    """
    max_unique_n = df.groupby('work_title')['author'].describe().unique.max()
    if raise_exception and max_unique_n > 1:
        raise ValueError(f"Must be one author per title!")
    else:
        return max_unique_n <= 1


def _fit_predict_for_pool(clf_train_test: tuple):
    """Вспомогательная функция для распараллеливания
    кросс-валидации.
    """
    clf, X_train, X_test, y_train, y_test = clf_train_test
    clf.fit(X_train, y_train)
    y_pred_test = pd.Series(clf.predict(X_test), index=y_test.index)
    return clf, y_pred_test


class CustomTfidfVectorizer:
    """Кастомный класс для получения tf-idf-матрицы из датафрейма
    с текстами и их лемматизированными версиями. Включает в себя
    два `TfidfVectorizer`: по символам в необработанных текстах и по леммам.
    """
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
    """Класс (пайплайн), совмещающий `CustomTfidfVectorizer` и
    линейный классификатор `SGDClassifier`.
    """
    def __init__(self, **kwargs):
        self.__sgd_args = None

        self.sgd_classifier_args = kwargs.get('sgd_classifier_args')

        estimators = [
            ("tfidf", CustomTfidfVectorizer(**kwargs)),
            ("clf", SGDClassifier(**self.sgd_classifier_args))
        ]
        self.pipeline = Pipeline(estimators)

    def fit(self, X, y=None, verbose=0):
        if y is None:
            X, y = X.drop('author', axis=1), X.author
        if not hasattr(X, 'lemmas'):
            ds.add_lemmas_column(X, inplace=True, verbose=verbose)
        check_work_titles_uniqueness(pd.concat([X, y], axis=1))
        self.pipeline.fit(X, y)
        return self

    def __predict_for_single_text(self, df_row):
        df = ds.make_dataset_of_excerpts(
            df=df_row,
            excerpt_num_of_words=EXCERPT_LEN
        )
        if df.shape[0] > 10:
            df = df.sample(10)
        elif df.shape[0] == 0:
            df = df_row
        ds.add_lemmas_column(df, inplace=True)
        predictions = self.pipeline.predict(df)
        if len(predictions) == 0:
            predictions.append(self.pipeline.predict(df_row))
        return Counter(predictions).most_common(1)[0][0]

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            predictions.append(self.__predict_for_single_text(row.to_frame().T))
        return np.array(predictions)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

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
