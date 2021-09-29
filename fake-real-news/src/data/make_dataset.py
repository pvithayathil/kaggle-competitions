from dotenv import find_dotenv, load_dotenv
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

load_dotenv(find_dotenv())
RAW_DATA_PATH = os.getcwd() + os.getenv("RAW_DATA_PATH")
PROCESSED_PATH = os.getcwd() + os.getenv("PROCESSED_PATH")

def main(input_filepath: str, output_filepath: str):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    Gets some features that's only possible from looking at the full data
    Eg. Is the Game a Back to Back Game?
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making data set from raw data")

    # Read Data
    fake_data = pd.read_csv(input_filepath + 'fake.csv')
    true_data = pd.read_csv(input_filepath + 'true.csv')

    # Tag Data
    fake_data['is_real']=0
    true_data['is_real']=1

    # Data Features
    data = pd.concat([fake_data, true_data], ignore_index=True)

    # Split Data
    X_data, Y_data = data.drop(["is_real"], axis=1), data["is_real"]
    x_train, x_val, y_train, y_val = train_test_split(X_data, Y_data, test_size=0.2)

    # Write Data Tables
    x_train.to_csv(output_filepath + "/train.csv", index=False)
    y_train.to_csv(output_filepath + "/train_target.csv", index=False)
    x_val.to_csv(output_filepath + "/test.csv", index=False)
    y_val.to_csv(output_filepath + "/test_target.csv", index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(RAW_DATA_PATH, PROCESSED_PATH)