from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use('ggplot')


def show_work_titles_histplot(df, title):
    sns.set(rc={'figure.figsize': (14, 4)})
    y = sns.histplot(data=df, x='author_surname')
    y.set(xlabel='', ylabel='Количество произведений')
    y.bar_label(y.containers[0])
    plt.title(title)
    max_num = df.author.value_counts().max()
    plt.yticks(np.arange(0, max_num+1, 5))
    plt.show()


def show_excerpts_histplot(df, title, num_of_words=None):
    sns.set(rc={'figure.figsize': (14, 4)})
    y = sns.histplot(data=df, x='author_surname')
    if num_of_words is None:
        sample_size = min(df.shape[0], 500)
        n_words_mean = df.sample(sample_size).text\
            .apply(lambda x: len(x.split())).mean()
        num_of_words = int(round(n_words_mean, -1))
    n_words_text = fr' по $\approx {num_of_words}$ слов'
    y.set(xlabel='', ylabel='Количество отрывков' + n_words_text)
    y.bar_label(y.containers[0])
    plt.title(title)
    plt.show()


def show_lemmas_histplot(df, title):
    sns.set(rc={'figure.figsize': (14,4)})
    hst = sns.histplot(
        data=df['lemmas'].apply(lambda x: len(x.split())),
    )
    hst.set(xlabel='Количество лемм в одном отрывке')
    hst.set(ylabel='Число отрывков')
    plt.title(title)
    plt.show()


def author_rus_surnames(df: pd.DataFrame) -> dict:
    """Возвращает словарь 'значение из столбца author (уник. для каждого автора)' -
    'фамилия автора на русском (из столбца author_surname)'.
    Возвращаемый словарь можно передать функции `plot_confusion_matrix`
    аргументом `rus_translation`.
    """
    return df.groupby('author')['author_surname'].describe().top.to_dict()


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, rus_translation=None):
    """Выводит матрицу ошибок в читаемом виде.
    """
    labels = y_true.unique()
    if rus_translation is not None:
        labels_rus = list(map(lambda x: rus_translation[x], labels))
    else:
        labels_rus = labels
    sns.set(rc={'figure.figsize': (5, 4)})
    sns.heatmap(
        confusion_matrix(y_true, y_pred, labels=labels),
        xticklabels=labels_rus,
        yticklabels=labels_rus,
        annot=True,
        fmt="d"
    )
    plt.show()


def false_predictions(df, y_pred) -> pd.DataFrame:
    """Возвращает DataFrame из объектов, для которых были сделаны
    ошибочные предсказания. Первые два столбца - ответы и предсказания.
    """
    y_true = df.author
    res = pd.DataFrame(np.vstack([y_true, y_pred]).T, columns=['y_true', 'y_pred'])
    res = pd.concat([res, df], axis=1)
    res = res.drop('author', axis=1)
    return res[res.y_true != res.y_pred]
