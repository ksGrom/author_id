import utils_tfidf_clf as u
import pytest
import pandas as pd
import numpy as np


def get_df_3():
    return pd.read_csv("./tests/test_files/train_sample_3.csv", index_col=0)


def test_check_work_titles_uniqueness():
    df = get_df_3()
    assert u.check_work_titles_uniqueness(df) is True
    df.loc[0, 'work_title'] = "story"
    df.loc[1, 'work_title'] = "story"
    df.loc[0, 'author'] = "author1"
    df.loc[1, 'author'] = "author2"
    assert u.check_work_titles_uniqueness(df, raise_exception=False) is False
    with pytest.raises(ValueError):
        u.check_work_titles_uniqueness(df)


class TestCustomTfidfVectorizer:
    def test_exceptions(self):
        df = get_df_3().reset_index(drop=True)
        tfidf = u.CustomTfidfVectorizer().fit(df)
        with pytest.raises(ValueError):
            tfidf.transform(df.drop('lemmas', axis=1))
        df.loc[0, 'lemmas'] = np.nan
        with pytest.raises(ValueError):
            tfidf.transform(df)


class TestAuthorIdentificationTfidfPipeline:
    def test_fit_predict(self):
        df = get_df_3().drop('lemmas', axis=1)
        pipe = u.AuthorIdentificationTfidfPipeline()
        pipe.fit(df)
        pred = pipe.predict(df)
        assert pred.shape == (3, )
        assert set(df.author) == set(pred)
