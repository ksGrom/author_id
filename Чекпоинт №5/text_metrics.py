
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
    foreign = []

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
            

            d[str(part_id) + ' ' + df['author'][part_id] + ' ' + df['work_title'][part_id]] = {**bs_data, **rs_data, 
                                                                                               **ms_data['gender'], **ms_data['number'], 
                                                                                               **ms_data['tense'], **ms_data['voice'], 
                                                                                               **ms_data['person'], **ds_data}
            df_new = df_new.append(d[str(part_id) + ' ' + df['author'][part_id] + ' ' + df['work_title'][part_id]], ignore_index=True)

            books.append(str(part_id) + ' ' + df['author'][part_id] + ' ' + df['work_title'][part_id])
            authors.append(df['author'][part_id])
            
            if re.sub("[^a-zA-Z]+", "", chapter_text) != '':
                foreign.append(1)
            else :
                foreign.append(0)
                
        except:
            pass
        
    df_new.index = books
    df_new['author'] = authors
    df_new['foreign'] = foreign
        
    return df_new

# функция, который приводит сырой датасет из метрик к тому виду, с которым работают модели
def prepare_metric_dataset(data, final_features):
    df = data.copy()
    for column in ['neut', 'masc', 'femn', 'plur','sing', 'pres', 'past', 
                 'futr', 'pssv', 'actv', '2per','3per', '1per']:
        df[column] = df[column] / df['n_words'] # делаем проценты вместо чисел для некоторых показателей
    df.index = df['Unnamed: 0'] # переопределяем индекс
    return df[final_features]

# функция, которая опреедляет переменные Х и у для трейна и теста
def get_variables(df_train, df_test):
    df_train1 = df_train.copy()
    df_test1 = df_test.copy()

    X_train = df_train1.drop(['author'], axis=1)
    y_train = df_train1['author']
    X_test = df_test1.drop(['author'], axis=1)
    y_test = df_test1['author']
    X_train.fillna(value=0, inplace=True)
    X_test.fillna(value=0, inplace=True)
#    X_train = X_train.set_index('Unnamed: 0')
#    X_test = X_test.set_index('Unnamed: 0')
    return X_train, y_train, X_test, y_test
