import os
from dotenv import find_dotenv, load_dotenv
import logging
import pandas as pd
import pickle
from sklearn.metrics import classification_report

load_dotenv(find_dotenv())
PROCESSED_PATH = os.getcwd() + os.getenv("PROCESSED_PATH")
READ_PATH_MODEL = PROCESSED_PATH + "/finalized_model.sav"


def read_model(input_path: str):
    """
    Args:
        input_path: path where the model is
    Returns:
        model:  current saved model

    """
    model = pickle.load(open(input_path, "rb"))

    return model


def read_features_data(input_path: str, is_train: bool) -> pd.DataFrame:

    """
    Args:
        input_path: location of processed data
        is_train: specifies if reading train or test

    Returns:
        data_matrix:  matrix of features ready for prediction
    """
    if is_train:
        data_matrix = pd.read_csv(input_path + "/train.csv")

    else:
        data_matrix = pd.read_csv(input_path + "/test.csv")
    print(data_matrix.head())
    print(data_matrix.shape)

    return data_matrix['clean_joined_title_and_text']


def read_target_data(input_path: str, is_train: bool) -> pd.DataFrame:

    """
    Args:
        input_path: location of processed data
        is_train: specifies if reading train or test

    Returns:
        df:  pre-processed dataframe ready to be made into features
    """
    if is_train:
        target = pd.read_csv(input_path + "/train_target.csv")

    else:
        target = pd.read_csv(input_path + "/test_target.csv")

    return target


def predict(model, features_data, target_data, output_path: str, is_train: bool):
    """

    Args:
        model: trained model ready to predict
        features_data: features data matrix
        target_data: target data
        is_train: is training or test data

    Returns:
        Side Effect, prints prediction score, writes predictions to csv
    """

    pred = model.predict(features_data)
    print(classification_report(target_data, pred))

    pred_df = pd.DataFrame({"is_fake_or_real": pred})

    if is_train:
        pred_df.to_csv(output_path + "/train_predictions.csv", index=False)
    else:
        pred_df.to_csv(output_path + "/test_predictions.csv", index=False)


def main(input_filepath, output_filepath, model_filepath):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("predict model")

    model = read_model(model_filepath)
    features_matrix = read_features_data(input_filepath, False)
    target = read_target_data(input_filepath, False)

    predict(model, features_matrix, target, output_filepath, False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main(PROCESSED_PATH, PROCESSED_PATH, READ_PATH_MODEL)