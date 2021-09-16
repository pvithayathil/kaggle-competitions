# from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

load_dotenv(find_dotenv())
RAW_DATA_PATH=os.getcwd()+os.getenv("RAW_DATA_PATH")
PROCESSED_PATH= os.getcwd()+os.getenv("PROCESSED_PATH")

def main(input_filepath: str, output_filepath: str):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    #Read Data
    data = pd.read_csv(input_filepath, parse_dates=['game_date'])

    # Data Cleansing
    data = data[data['shot_made_flag'].notna()]

    # Split Data
    X_data, Y_data = data.drop('shot_made_flag', axis=1), data['shot_made_flag']
    x_train, x_val, y_train, y_val = train_test_split(X_data, Y_data, test_size=.2)

    # Write Data Tables
    x_train.to_csv(output_filepath + '/train.csv',index=False)
    y_train.to_csv(output_filepath + '/train_target.csv',index=False)
    x_val.to_csv(output_filepath + '/test.csv',index=False)
    y_val.to_csv(output_filepath + '/test_target.csv', index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main(RAW_DATA_PATH, PROCESSED_PATH)