from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np

import utils_dataset as ds

EXCERPT_LEN = 250
def empty_logger(x): return


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

    def fit(self, X, y=None,  *, preprocessing=True):
        self.__fit_transform(X, 'fit', preprocessing)
        return self

    def transform(self, X, y=None,  *, preprocessing=True):
        return self.__fit_transform(X, 'transform', preprocessing)

    def fit_transform(self, X, y=None, *, preprocessing=True):
        return self.__fit_transform(X, 'fit_transform', preprocessing)

    def __fit_transform(self, X, method_name, preprocessing):
        X_text = self.__get_text_series(X)
        if preprocessing:
            X_text = X_text.apply(self.__text_preprocessing)
        X_text_res = getattr(self.char_tfidf_vectorizer, method_name)(X_text)
        return X_text_res

    def __get_text_series(self, df):
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
        self.__patterns = r'[^а-яa-z.?!…,:;—()\"\'\- ]+' \
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
            'max_features': 30000,
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

        self.vectorizer = CustomTfidfVectorizer(**kwargs)
        self.classifier = SGDClassifier(**self.sgd_classifier_args)
        self.__n_fit = 0  # количество вызовов `fit`

    def fit(self, X, y=None, *, preprocessing=True, logger=empty_logger):
        if y is None:
            X, y = X.drop('author', axis=1), X.author
        logger("FEATURE EXTRACTION")
        if self.__n_fit == 0:
            X = self.vectorizer.fit_transform(X, preprocessing=preprocessing)
        else:
            X = self.vectorizer.transform(X, preprocessing=preprocessing)
        logger("TRAINING")
        self.classifier.fit(X, y)
        self.__n_fit += 1
        return self

    def predict(self, X, *, preprocessing=True, logger=empty_logger):
        predictions = []
        logger("PREPROCESSING")
        X = ds.make_dataset_of_excerpts(
            df=X,
            excerpt_num_of_words=EXCERPT_LEN
        )
        logger("PREDICTING")
        for orig_text_id in X.orig_text_id.unique():
            X_single_text_excerpts = X[X.orig_text_id == orig_text_id]
            if X_single_text_excerpts.shape[0] > 20:
                X_single_text_excerpts = X_single_text_excerpts.sample(20)
            X_single_text_excerpts = self.vectorizer.transform(
                X_single_text_excerpts, preprocessing=preprocessing)
            predictions.append(
                pd.Series(self.classifier.predict(X_single_text_excerpts))
                .mode().tolist()[0]
            )
        return np.array(predictions)

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
            'loss': 'modified_huber',
            'warm_start': True
        }
        if type(value) is dict:
            self.__sgd_args.update(value)
