from argparse import ArgumentParser
import joblib
import nltk
from dataset_utils import (
    input_file_to_df, 
    add_lemmas_column
)

nltk.download('stopwords')


def parse_args():
    parser = ArgumentParser('Text classification')
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to txt file with one text or csv file with several texts.'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Path to output csv file with predictions.'
    )
    return parser.parse_args()


def main(args):
    with open(args.input, 'r') as f:
        df = input_file_to_df(f)
    df = add_lemmas_column(df)
    clf = joblib.load("default.pkl")
    df['predictions'] = clf.predict(df)
    df.to_csv(args.output)


if __name__ == '__main__':
    args = parse_args()
    main(args)
