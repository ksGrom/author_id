from pymorphy2 import MorphAnalyzer
from collections import namedtuple
from nltk.corpus import stopwords
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import warnings
import glob
import re
import os


morph = MorphAnalyzer()


def _replace_punctuation_marks(text):
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


def df_from_txt_files(dataset_name, dir_path="./texts"):
    """Создает датафрейм с сырыми данными из текстовых файлов.
    Столбцы: автор (по названию папки), фамилия автора на русском
    (из файла AUTHOR.txt в каждой папке), название произведения, текст произведения.
    Тексты должны располагаться в директориях с адресом вида
    {`dir_path`}/{`dataset_name`}/ИМЯ_АВТОРА
    (для каждого автора - отдельная папка).
    В качестве `dataset_name` можно указать TRAIN, TEST и др.

    Parameters
    ----------
    dataset_name : str
        Название датасета и часть пути, по которому ищутся тексты.

    dir_path : str, default="./texts"
        Путь к директории с датасетами.
    """

    # Находим все текстовые файлы
    file_paths = glob.glob(
        os.path.join(dir_path, f'{dataset_name}/**/*.txt'),
        recursive=True
    )

    if len(file_paths) == 0:
        raise FileNotFoundError

    # Находим файлы с информацией о каждом авторе
    # (AUTHOR.txt, сейчас там только фамилия на русском)
    author_info_paths = glob.glob(
        os.path.join(dir_path, f'{dataset_name}/**/AUTHOR.txt'),
        recursive=True
    )

    # Создаем словарь "автор" (название папки) - "фамилия"
    # (из файла AUTHOR.txt в этой папке)
    author__rus_surname__dict = dict()
    for path in author_info_paths:
        with open(path, 'r', encoding='utf-8') as f:
            author__rus_surname__dict[path.split("\\")[-2]] \
                = f.readline().split(':')[-1]

    # Создаем словарь "название произведения" (по названию файла) - "путь к файлу"
    title_path_dict = {path.split("\\")[-1].split(".")[0]
                       : path for path in file_paths}
    del title_path_dict['AUTHOR']
    # Создаем словарь "название произведения" - "автор"
    title_author_dict = {path.split("\\")[-1].split(".")[0]
                         : path.split("\\")[-2] for path in file_paths}
    del title_author_dict['AUTHOR']

    # Создаем датафрейм и сохраняем
    dataset_list = []
    for title, path in title_path_dict.items():
        with open(path, 'r', encoding='utf-8') as f:
            text = _replace_punctuation_marks(f.read()[:-2000])
        author = title_author_dict[title]
        dataset_list.append({
            'author': author,
            'author_surname': author__rus_surname__dict[author],
            'work_title': title,
            'text': text
        })
    df_raw = pd.DataFrame(dataset_list)
    return df_raw


def sentence_generator(text, offset=0, punct_marks=".?!…", yield_pos=False):
    """Генератор предложений. Отдает по одному предложению
    из входного текста (`text`) за итерацию.
    """
    punct_marks = set(punct_marks)
    while offset < len(text) and not text[offset].isalpha():
        offset += 1
    start_pos = i = offset
    while i < len(text):
        if text[i] in punct_marks:
            while i < len(text) and not text[i].isalpha():
                i += 1
            end_pos = i
            if not yield_pos:
                yield text[start_pos:end_pos].strip()
            else:
                yield start_pos, end_pos
            start_pos = end_pos
        i += 1


def excerpt_generator(text, excerpt_len, offset_n_words='excerpt_len'):
    """Генератор отрывков текста. Возвращает по одному отрывку за итерацию.
    Отрывок подбирается так, чтобы первое и последнее предложения в нем
    входили в него целиком.

    Parameters
    ----------
    text : str
        Обрабатываемый текст.

    excerpt_len : int
        Минимальная длина отрывка в словах. Если текст меньше данной длины,
        то не будет сгенерировано ни одного отрывка.

    offset_n_words : int, default='excerpt_len'
        Отступ в словах начала очередного отрывка от начала предыдущего отрывка.
        Если `'excerpt_len'` (по-умолчанию), то устанавливается равным длине
        отрывка; в этом случае отрывки не пересекаются.
    """
    if type(offset_n_words) is not int:
        offset_n_words = excerpt_len
    offset = 0
    stop = False
    while not stop:
        excerpt = ''
        for sentence in sentence_generator(text, offset):
            excerpt += (' ' + sentence)
            if len(excerpt.split()) >= excerpt_len:
                yield excerpt.strip()
                break

        stop = True
        offset_excerpt_len = 0
        for start_pos, end_pos in sentence_generator(text, offset, yield_pos=True):
            offset_excerpt_len += len(text[start_pos:end_pos].split())
            if offset_excerpt_len >= offset_n_words:
                offset += len(text[offset:end_pos])
                stop = False
                break


def _open_tqdm(verbose, total):
    if verbose != 0:
        pbar = tqdm(total=total)
    else:
        PbarBlank = namedtuple('pbar', ['update', 'close'])
        def do_nothing(*args, **kwargs): return
        pbar = PbarBlank(update=do_nothing, close=do_nothing)
    return pbar


def _tqdm(iterator, *, total, verbose):
    if verbose != 0:
        return tqdm(iterator, total=total)
    else:
        return iterator


def make_dataset_of_excerpts(df, excerpt_num_of_words=250,
                             offset='excerpt_len', verbose=1):
    """Из датафрейма с текстами создает датафрейм с отрывками
    по не менее чем `excerpt_num_of_words` слов. Каждый отрывок подбирается так,
    чтобы первое и последнее предложения в нем входили в него целиком.

    Parameters
    ----------
    df : pd.DataFrame
        Обрабатываемый датафрейм.

    excerpt_num_of_words : int, default=250
        Минимальная длина отрывка в словах. Если текст меньше данной длины,
        то не будет сгенерировано ни одного отрывка.

    offset : int, default='excerpt_len'
        Отступ в словах начала очередного отрывка от начала предыдущего отрывка.
        Если `'excerpt_len'` (по-умолчанию), то устанавливается равным длине
        отрывка; в этом случае отрывки не пересекаются.

    verbose : {0, 1}, default=1
        Если 1, то показывает полосу прогресса (tqdm).
    """
    dataset_list = []
    pbar = _open_tqdm(verbose, df.shape[0])
    for index, row in df.iterrows():
        excerpts = list(excerpt_generator(row['text'], excerpt_num_of_words, offset))
        for i in range(len(excerpts)):
            excerpt = excerpts[i]
            dataset_list.append({
                'author': row['author'],
                'author_surname': row['author_surname'],
                'work_title': row['work_title'],
                'excerpt_num': i,
                'text': excerpt
            })
        pbar.update(1)
    pbar.close()
    return pd.DataFrame(dataset_list)


def lemmatize(doc: str, remove_stop_words=True, patterns=r'[^а-яё ]+') -> list[str]:
    """Лемматизация текста. Регистр не учитывается.
    """
    doc = re.sub(patterns, ' ', doc.lower()).strip()
    tokens = []
    stopwords_ru = stopwords.words("russian") if remove_stop_words else ''
    for token in doc.split():
        if token and token.strip() not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            tokens.append(token)
    return tokens


def _lemmatize_row_for_pool(row):
    return ' '.join(lemmatize(row))


def add_lemmas_column(df, inplace=False, verbose=1, n_jobs=6):
    """Лемматизирует тексты (столбец text в `df`), добавляет
    в исходный датафрейм столбец с леммами.
    """
    if not inplace:
        df = df.copy(deep=True)
    with Pool(n_jobs) as p:
        lemmas_ = _tqdm(
            p.imap(
                _lemmatize_row_for_pool,
                df['text'].to_list()
            ),
            verbose=verbose,
            total=df.shape[0]
        )
        lemmas_list = list(lemmas_)
    df['lemmas'] = pd.Series(lemmas_list, index=df.index)
    if df.lemmas.isna().sum() > 0 or (df.lemmas.values == '').sum() > 0:
        warnings.warn(
            "Не все тексты удалось лемматизировать. Возможно, они полностью "
            "состоят из стоп-слов или написаны латиницей. Тексты с NaN-значениями "
            "в столбце lemmas были удалены."
        )
        df.drop(df[df.lemmas.isna()].index, inplace=True)
        df.drop(df[df.lemmas.values == ''].index, inplace=True)
    return df


def undersampling(df):
    res_df = df.copy(deep=True)
    min_count = df.author.value_counts().min()
    for author in df.author.unique():
        n_drop = df.author.value_counts()[author] - min_count
        idx_drop = res_df[res_df.author == author].sample(n_drop).index
        res_df.drop(idx_drop, inplace=True)
    return res_df


def df_leave_authors(df, authors):
    mask = None
    for author in authors:
        if mask is None:
            mask = (df.author == author)
        else:
            mask = mask | (df.author == author)
    return df[mask].copy(deep=True)
