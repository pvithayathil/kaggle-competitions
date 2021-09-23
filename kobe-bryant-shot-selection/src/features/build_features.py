import os
from dotenv import find_dotenv, load_dotenv
import logging
import numpy as np
import pandas as pd
from scipy import sparse
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
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
    # Create Loc X,Y Binned Data:
    # X axis between [-250, 250] and
    # Y axis between [-50, 800]
    loc_x_bin = [-251, -200, -150, -100, -50, 0,
                 50, 100, 150, 200, 251
                 ]
    loc_x_bin_labels = [-250, -200, -150, -100, -50, 0,
                        50, 100, 150, 200
                        ]

    loc_y_bin = [-50, 0, 50, 100, 150,
                 200, 250, 300, 350,
                 400, 450, 500, 550,
                 800
                 ]
    loc_y_bin_labels = [-50, 0, 50, 100, 150,
                        200, 250, 300, 350,
                        400, 450, 500, 550,
                        ]
    # Loc_x, and loc_y binning
    df['loc_x_cut'] = pd.cut(df['loc_x'], loc_x_bin, labels=loc_x_bin_labels)
    df['loc_y_cut'] = pd.cut(df['loc_y'], loc_y_bin, labels=loc_y_bin_labels)
    df['loc_bin'] = df['loc_x_cut'].astype(str) + "_" + df['loc_y_cut'].astype(str)

    # Create Date Columns
    df['season'] = pd.to_datetime(df['season'].str[:4], infer_datetime_format=True)
    df['game_month'] = df['game_date'].dt.month

    # Feature Scored Lots of Points in Game
    df['made_over_25_points_in_shots'] = np.select([df['num_points_made_in_game'] >= 25],
                                                         [1],
                                                         default=0)
    # FG % in the game @ time of the shot
    df["field_goal_pct_in_game"] = 1.0 * df['num_shots_made_in_game'] / df['num_shots_taken_in_game']
    df["field_goal_pct_in_game"].fillna(0, inplace=True)

    cat_vars = [
        "action_type",
        "period",
        "playoffs",
        "season",
        "opponent",
        "shot_type",
        "num_shots_made_in_last_five_attempts",
        "num_game_last_7_days",
        "game_month",
        "loc_bin"
    ]

    num_vars = [
        "shot_distance",
        "num_shots_taken_in_game",
        "field_goal_pct_in_game",
        "shot_is_in_possibly_clutch_moment",
        "is_home_game",
        "game_is_back_to_back",
        "made_over_25_points_in_shots"
    ]

    df_features = df[cat_vars+num_vars]
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
    ohe = OneHotEncoder(handle_unknown = 'ignore')

    si_1 = SimpleImputer(strategy='mean')
    mms = MinMaxScaler()

    # define column groups with same processing
    cat_vars = [
        "action_type",
        "period",
        "playoffs",
        "season",
        "opponent",
        "shot_type",
        "num_shots_made_in_last_five_attempts",
        "num_game_last_7_days",
        "game_month",
        "loc_bin"
    ]

    num_vars = [
        "shot_distance",
        "num_shots_taken_in_game",
        "field_goal_pct_in_game",
        "shot_is_in_possibly_clutch_moment",
        "is_home_game",
        "game_is_back_to_back",
        "made_over_25_points_in_shots"
    ]
    # set up pipelines for each column group
    categorical_pipe = Pipeline([("si_0", si_0), ("ohe", ohe)])
    numerical_pipe = Pipeline([("si_1", si_1), ("mms", mms)])

    # set up columnTransformer
    col_transformer = ColumnTransformer(
        transformers=[
            ("cats", categorical_pipe, cat_vars),
            ('nums', numerical_pipe, num_vars),
        ],
        remainder="drop", n_jobs=-1
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
