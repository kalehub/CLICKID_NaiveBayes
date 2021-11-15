import numpy as np
import pandas as pd


def preprocess_data(data):
    '''
    Preprocess general data:
    Removing duplicate value and one hot encoding

    '''
    # check jika ada data yang terduplikasi (ada 10 data)
    data = data.drop_duplicates(subset='title', keep=False)
    # drop kolom 'label_score' untuk one hot encoding
    data = data.drop(columns=['label_score'])
    # one hot encoding
    one_hot = pd.get_dummies(data['label'])
    data = data.drop(columns=['label'])
    data = data.join(one_hot)
    return data


def main():
    DATASET_DIR = '/Users/teguhsatya/Projects/CLICKID/dataset/archive/annotated/combined/csv/all_agree.csv'
    df = pd.read_csv(DATASET_DIR)
    df_clean = preprocess_data(df)
    print(df_clean.head())
    

if __name__ == '__main__':
    main()
