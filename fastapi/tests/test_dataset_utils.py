import dataset_utils as u
import pytest
import os
from pathlib import Path
import random
import pandas as pd
import warnings


def new_dir(path):
    path = Path(path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def random_text(length):
    letters = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    letters += letters.upper() + "1234567890,!?.—-" + 30 * " "
    return ''.join(random.choice(letters) for _ in range(length))


def new_file(path):
    path = Path(path)
    with open(path, 'a') as f:
        f.write(random_text(random.randint(10**3, 10**4)))
    return path


def make_df():
    return pd.DataFrame([
        {'author': 'first', 'text': 'Это первый текст первого автора.'},
        {'author': 'first', 'text': 'Это второй текст первого автора.'},
        {'author': 'second', 'text': 'Это первый текст второго автора.'},
        {'author': 'second', 'text': 'Это второй текст второго автора.'},
        {'author': 'third', 'text': 'Это единственный текст третьего автора.'}
    ])


def make_lemmas_series():
    return pd.Series([
        'это первый текст первый автор',
        'это второй текст первый автор',
        'это первый текст второй автор',
        'это второй текст второй автор',
        'это единственный текст третий автор'
    ])


@pytest.fixture(scope="session")
def tmp_folder(tmp_path_factory):
    tmp = tmp_path_factory.mktemp(".test_tmp")
    new_dir(os.path.join(tmp, 'texts'))

    new_dir(os.path.join(tmp, 'texts/TRAIN'))
    new_dir(os.path.join(tmp, 'texts/TRAIN/AUTHOR_1'))
    new_file(os.path.join(tmp, 'texts/TRAIN/AUTHOR_1/story.txt'))
    new_file(os.path.join(tmp, 'texts/TRAIN/AUTHOR_1/short_story.txt'))
    new_dir(os.path.join(tmp, 'texts/TRAIN/AUTHOR_2'))
    new_file(os.path.join(tmp, 'texts/TRAIN/AUTHOR_2/novella.txt'))

    new_dir(os.path.join(tmp, 'texts/TEST'))
    new_dir(os.path.join(tmp, 'texts/TEST/AUTHOR_1'))
    new_file(os.path.join(tmp, 'texts/TEST/AUTHOR_1/novel.txt'))
    new_dir(os.path.join(tmp, 'texts/TEST/AUTHOR_2'))
    new_file(os.path.join(tmp, 'texts/TEST/AUTHOR_2/tale.txt'))
    new_file(os.path.join(tmp, 'texts/TEST/AUTHOR_2/poem.txt'))
    return tmp


@pytest.mark.parametrize(
    ['orig', 'res'], [
        ('Привет...', 'Привет…'),
        ('музей-усадьба «Ясная Поляна»', 'музей-усадьба "Ясная Поляна"'),
        ('Текст.\nТекст.', 'Текст. Текст.'),
    ]
)
def test__replace_punctuation_marks(orig, res):
    assert res == u._replace_punctuation_marks(orig)


def test_df_from_txt_files():
    pass


def test_df_from_txt_files_empty(tmp_folder):
    with pytest.raises(FileNotFoundError):
        u.df_from_txt_files(
            "AAA", dir_path=os.path.join(tmp_folder, 'texts'))
    with pytest.raises(FileNotFoundError):
        u.df_from_txt_files(
            "TRAIN", dir_path=os.path.join(tmp_folder, 'texts'))


@pytest.mark.parametrize(
    ['doc', 'remove_stop_words', 'result'], [
        ('Hello Привет! Hello. Как дела?', False, ['привет', 'как', 'дело']),
        ('Hello Привет! Hello. Как дела?', True, ['привет', 'дело']),
        ('Много людей.', True, ['человек']),
    ]
)
def test_lemmatize(doc, remove_stop_words, result):
    assert tuple(result) == tuple(u.lemmatize(doc, remove_stop_words))


def test__lemmatize_row_for_pool():
    assert "привет дело" == u._lemmatize_row_for_pool("Привет! hi. Как дела?")


@pytest.mark.parametrize(
    ['n_jobs'], [[1], [2]]
)
def test_add_lemmas_column(n_jobs):
    orig_df = make_df()
    df = orig_df.copy()
    res_df = u.add_lemmas_column(df, inplace=False, n_jobs=n_jobs)
    assert make_lemmas_series().to_list() == res_df.lemmas.to_list()
    assert df.to_dict() == orig_df.to_dict()
    assert df.to_dict() != res_df.to_dict()

    u.add_lemmas_column(df, inplace=True, n_jobs=n_jobs)
    assert df.to_dict() != orig_df.to_dict()
    assert df.to_dict() == res_df.to_dict()


def test_add_lemmas_column_warning():
    warnings.filterwarnings("error")
    df = make_df()
    df.iloc[0].text = "english"
    with pytest.raises(UserWarning):
        u.add_lemmas_column(df, inplace=True)
    warnings.filterwarnings("default")


def test_undersampling():
    df = u.undersampling(make_df())
    assert (3, 2) == df.shape
    for author in df.author:
        assert (1, 2) == df[df.author == author].shape


def test_df_leave_authors():
    df = u.df_leave_authors(make_df(), ["first", "third"])
    assert (3, 2) == df.shape
    assert ["first", "third"] == df.author.unique().tolist()


@pytest.mark.parametrize(
    ['filename', 'shape'], [
        ['text_example.txt', (1, 1)],
        ['train_sample_3.csv', (3, 7)],
    ]
)
def test_input_file_to_df(filename, shape):
    file = open(f'./tests/test_files/{filename}', 'br')
    df = u.input_file_to_df(file)
    assert 'text' in df.columns
    assert shape == df.shape
    file.close()


@pytest.mark.parametrize(
    ['filename'], [
        ['cp1251_example.txt'],
        ['invalid.jpg'],
        ['no_text_column.csv']
    ]
)
def test_input_file_to_df_exception(filename):
    file = open(f'./tests/test_files/{filename}', 'br')
    with pytest.raises(ValueError):
        u.input_file_to_df(file)
    file.close()
