from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from scipy import stats
import pandas as pd
import numpy as np

import utils_dataset as ds

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


class CustomTfidfVectorizer:
    """Кастомный класс для получения tf-idf-матрицы из датафрейма
    с текстами. Включает в себя посимвольный `TfidfVectorizer`.
    """

    def __init__(self, **kwargs):
        self.__char_tfidf_params = None
        self.__patterns = None

        self.char_tfidf_params = kwargs.get('char_tfidf_params')
        self.patterns = kwargs.get('patterns')

        self.char_tfidf_vectorizer = TfidfVectorizer(**self.char_tfidf_params)

    def fit(self, X, y=None):
        self.__fit_transform(X, 'fit')
        return self

    def transform(self, X, y=None):
        return self.__fit_transform(X, 'transform')

    def fit_transform(self, X, y=None):
        return self.__fit_transform(X, 'fit_transform')

    def __fit_transform(self, X, method_name):
        X_text = self.__get_text_and_lemmas_series(X)
        X_text = X_text.apply(self.__text_preprocessing)
        X_text_res = getattr(self.char_tfidf_vectorizer, method_name)(X_text)
        return X_text_res

    def __get_text_and_lemmas_series(self, df):
        if not (type(df) is pd.DataFrame
                and hasattr(df, 'text')):
            err = "X must be pandas.DataFrame with `text` column!"
            raise ValueError(err)
        text_series = df.text
        if text_series.isna().sum() > 0:
            raise ValueError("Nan value(s) in `text` column!")
        return text_series

    def __text_preprocessing(self, text):
        return ds.text_preprocessing(
            text, lower=True, patterns=self.patterns)

    @property
    def patterns(self):
        return self.__patterns

    @patterns.setter
    def patterns(self, value):
        self.__patterns = r'[^а-я.?!…,:;—()\"\'\- ]+' \
            if value is None else value

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


class AuthorIdTfidfPipeline:
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
        check_work_titles_uniqueness(pd.concat([X, y], axis=1))
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        predictions = []
        X = ds.make_dataset_of_excerpts(
            df=X,
            excerpt_num_of_words=EXCERPT_LEN
        )
        for text_id in X.text_id.unique():
            X_single_text_excerpts = X[X.text_id == text_id]
            if X_single_text_excerpts.shape[0] > 20:
                X_single_text_excerpts = X_single_text_excerpts.sample(20)
            predictions.append(
                pd.Series(self.pipeline.predict(X_single_text_excerpts)).mode().tolist()[0]
            )
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
            'max_iter': 5000,
            'verbose': 0,
            'n_jobs': -1,
            'loss': 'modified_huber'
        }
        if type(value) is dict:
            self.__sgd_args.update(value)
