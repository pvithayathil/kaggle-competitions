import os
from dotenv import find_dotenv, load_dotenv
import logging
import numpy as np
import pandas as pd
import nltk
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

load_dotenv(find_dotenv())
PROCESSED_PATH = os.getcwd() + os.getenv("PROCESSED_PATH")
nltk.download('stopwords')

STOP_WORDS = stopwords.words('english')
STOP_WORDS.extend(['from', 'subject', 're', 'edu', 'use'])

def read_preprocessed_data(input_path: str, is_train: bool) -> pd.DataFrame:
    """

    Args:
        input_path: location of processed data
        is_train: specifies if reading train or test

    Returns:
        df:  pre-processed dataframe ready to be made into features
    """
    if is_train:
        df = pd.read_csv(input_path + "/train.csv")
    else:
        df = pd.read_csv(input_path + "/test.csv")

    return df

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2 and token not in STOP_WORDS:
            result.append(token)

    return result

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        df: either train or test
    Returns:
        df: dataframe that has features
    """
    # remove stop words
    df['title_and_text'] = df['title'] + ' ' + df['text']
    df['clean_title_and_text'] = df['title_and_text'].apply(preprocess)
    df['clean_joined_title_and_text'] = df['clean_title_and_text'].apply(lambda x: " ".join(x))


    df['clean_title'] = df['title'].apply(preprocess)
    df['clean_joined_title'] = df['clean_title'].apply(lambda x: " ".join(x))

    df['clean_text'] = df['text'].apply(preprocess)
    df['clean_joined_text'] = df['clean_text'].apply(lambda x: " ".join(x))


    return df

def main(input_filepath, output_filepath):
    """"""
    logger = logging.getLogger(__name__)
    logger.info("making features from  training data set from processed data")

    # Read Data
    df_train = read_preprocessed_data(input_filepath, True)
    df_test = read_preprocessed_data(input_filepath, False)

    df_train = make_features(df_train)
    df_test = make_features(df_test)

    print(df_train.head())
    print(df_test.head())

    #train, test, feature_names = make_datasets_model_ready(df_train, df_test)

    df_train.to_csv(output_filepath + "/train.csv")
    df_test.to_csv(output_filepath + "/test.csv")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(PROCESSED_PATH, PROCESSED_PATH)