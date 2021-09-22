import os
from dotenv import find_dotenv, load_dotenv
import logging
import numpy as np
import pandas as pd
from scipy import sparse
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

load_dotenv(find_dotenv())
PROCESSED_PATH = os.getcwd() + os.getenv("PROCESSED_PATH")


def read_preprocessed_data(input_path: str, is_train: bool) -> pd.DataFrame:
    """

    Args:
        input_path: location of processed data
        is_train: specifies if reading train or test

    Returns:
        df:  pre-processed dataframe ready to be made into features
    """
    if is_train:
        df = pd.read_csv(input_path + "/train.csv", parse_dates=["game_date"])
    else:
        df = pd.read_csv(input_path + "/test.csv", parse_dates=["game_date"])

    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """

    Args:
        df: either train or test
    Returns:
        df_features: dataframe that has features
    """
    feature_cols = [
        "combined_shot_type",
        "period",
        "season",
        "shot_zone_basic",
        "shot_zone_range",
    ]
    df_features = df[feature_cols]
    return df_features


def make_datasets_model_ready(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> (sparse.csr_matrix, sparse.csr_matrix, list):
    """

    Args:
        df_train: training data frame
        df_test: test data frame

    Returns:
        X_train_transformed: matrix for training features
        X_test_transformed: matrix for training features
        col_names: list of feature column names

    """

    # define transformers
    si_0 = SimpleImputer(strategy="constant", fill_value="missing")
    ohe = OneHotEncoder()
    # define column groups with same processing
    cat_vars = [
        "combined_shot_type",
        "period",
        "season",
        "shot_zone_basic",
        "shot_zone_range",
    ]
    # set up pipelines for each column group
    categorical_pipe = Pipeline([("si_0", si_0), ("ohe", ohe)])
    # set up columnTransformer
    col_transformer = ColumnTransformer(
        transformers=[("cats", categorical_pipe, cat_vars)], remainder="drop", n_jobs=-1
    )

    X_train_transformed = col_transformer.fit_transform(df_train)
    X_test_transformed = col_transformer.transform(df_test)

    col_names = (
        col_transformer.named_transformers_["cats"]
        .named_steps["ohe"]
        .get_feature_names()
    )

    return X_train_transformed, X_test_transformed, col_names


def main(input_filepath, output_filepath):
    """"""
    logger = logging.getLogger(__name__)
    logger.info("making features from  training data set from processed data")

    # Read Data
    df_train = read_preprocessed_data(input_filepath, True)
    df_test = read_preprocessed_data(input_filepath, False)

    df_train = make_features(df_train)
    df_test = make_features(df_test)

    train, test, feature_names = make_datasets_model_ready(df_train, df_test)

    sparse.save_npz(output_filepath + "/train.npz", train)
    sparse.save_npz(output_filepath + "/test.npz", test)

    with open(output_filepath + "/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(PROCESSED_PATH, PROCESSED_PATH)
