
# функция, которая из датасета с текстами выдает датасет с метриками для этих текстов
def get_metrics_dataset(df):

    from ruts import BasicStats
    from ruts import ReadabilityStats
    from ruts import DiversityStats
    from ruts import MorphStats
    from tqdm import tqdm
    import pandas as pd

    books = []
    authors = []

    df_new = pd.DataFrame()
    d = {}

    for part_id in tqdm(range(len(df['text']))):
        chapter_text = df['text'][part_id].strip().replace('\n', '')

        try:
            # добавляем базовые статистики
            bs = BasicStats(chapter_text)
            bs_data = dict(bs.get_stats())
            del bs_data['c_letters']
            del bs_data['c_syllables']


            # метрики читаемости
            rs = ReadabilityStats(chapter_text)
            rs_data = dict(rs.get_stats())


            # морфологические метрики
            ms = MorphStats(chapter_text)
            ms_data = dict(ms.get_stats())


            # метрики лексического разнообразия
            ds = DiversityStats(chapter_text)
            ds_data = dict(ds.get_stats())

            d[str(part_id) + ' ' + df['author'][part_id] + ' ' + df['work_title'][part_id]] = {**bs_data, **rs_data, **ms_data['gender'], **ms_data['number'], **ms_data['tense'], **ms_data['voice'], **ms_data['person'], **ds_data}
            df_new = df_new.append(d[str(part_id) + ' ' + df['author'][part_id] + ' ' + df['work_title'][part_id]], ignore_index=True)

            books.append(str(part_id) + ' ' + df['author'][part_id] + ' ' + df['work_title'][part_id])
            authors.append(df['author'][part_id])
            df_new.index = books
            df_new['author'] = authors
            df_new.to_csv('TEST_500words_lemmatized_metrics.csv')
        except:
            pass

# функция, который приводит сырой датасет из метрик к тому виду, с которым работают модели
def prepare_metric_dataset(df):
    for column in ['neut', 'masc', 'femn', 'plur','sing', 'pres', 'past', 
                 'futr', 'pssv', 'actv', '2per','3per', '1per']:
        df[column] = df[column] / df['n_words'] # делаем проценты вместо чисел для некоторых показателей
    df.fillna(value=0, inplace=True)
    df.index = df['Unnamed: 0'] # переопределяем индекс
    df.drop(['n_sents', 'n_words', 'n_unique_words', 'n_long_words',
            'n_complex_words', 'n_simple_words', 'n_monosyllable_words',
            'n_polysyllable_words', 'n_chars', 'n_letters', 'n_spaces',
            'n_syllables', 'n_punctuations', 'Unnamed: 22', 'Unnamed: 0', 'flesch_kincaid_grade', 'flesch_reading_easy', 'coleman_liau_index', 'automated_readability_index',
                    'ttr', 'rttr', 'cttr', 'httr', 'sttr', 'dttr', 'mattr', 'msttr', 'mamtld', 'hapax_index', 'mtld', 
                    'simpson_index', 'lix'], axis=1, inplace=True)  # удаляем метрики, которые зависят от длины или очень похожи
    return df

# функция, которая опреедляет переменные Х и у для трейна и теста
def get_variables(df_train, df_test):
    df_train1 = df_train.copy()
    df_test1 = df_test.copy()

    df_test1 = prepare_metric_dataset(df_test1)
    X_train = df_train1.drop(['author', 'author_num'], axis=1)
    y_train = df_train1['author']
    X_test = df_test1.drop(['author'], axis=1).reindex(columns=X_train.columns)
    y_test = df_test1['author']
    X_train = X_train.set_index('Unnamed: 0')
    X_test = X_test.set_index('Unnamed: 0')
    return X_train, y_train, X_test, y_test

def logreg_for_metrics(df_train, df_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    # df_train1 = prepare_metric_dataset(df_train)
    # df_test1 = prepare_metric_dataset(df_test)
    df_train = df_train.fillna(value=0)
    df_test = df_test.fillna(value=0)
    X_train, y_train, X_test, y_test = get_variables(df_train, df_test)

    logreg_model = LogisticRegression(multi_class='multinomial', max_iter=10000, C=121) # уже подобранные гиперпараметры
    logreg_model.fit(X_train, y_train)
    pred_logreg = logreg_model.predict(X_test)
    print('F1-score: ', f1_score(y_test, pred_logreg, average='micro'))

    return logreg_model

def catboost_for_metrics(df_train, df_test):
    from catboost import CatBoostClassifier
    from sklearn.metrics import f1_score
    # df_train1 = prepare_metric_dataset(df_train)
    # df_test1 = prepare_metric_dataset(df_test)
    X_train, y_train, X_test, y_test = get_variables(df_train, df_test)

    model = CatBoostClassifier()
    model.fit(X_train, y_train, silent=True)
    preds = model.predict(X_test)
    print('F1-score: ', f1_score(preds, y_test, average='micro'))
    return model